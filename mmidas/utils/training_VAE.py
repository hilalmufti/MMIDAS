import torch
import pickle
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.metrics.cluster import adjusted_rand_score
import torch.nn.utils.prune as prune
import time, glob
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import linear_sum_assignment
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
from sklearn.model_selection import train_test_split
from utils.augmentation.udagan import *
from utils.vae import VAE


class train_VAE:

    def __init__(self, saving_folder='', device=None, eps=1e-8, save_flag=True):

        self.eps = eps
        self.save = save_flag
        self.folder = saving_folder
        self.device = device

        if device is None:
            self.gpu = False
            print('using CPU ...')
        else:
            self.gpu = True
            torch.cuda.set_device(device)
            gpu_device = torch.device('cuda:' + str(device))
            print('using GPU ' + torch.cuda.get_device_name(torch.cuda.current_device()))


    def data_gen(self, dataset, train_size):

        test_size = dataset.shape[0] - train_size
        train_cpm, test_cpm, train_ind, test_ind = train_test_split(
            dataset[:, self.index], np.arange(dataset.shape[0]), train_size=train_size, test_size=test_size, random_state=0)

        train_cpm, val_cpm, train_ind, val_ind = train_test_split(train_cpm, train_ind, train_size=train_size - test_size, test_size=test_size, random_state=0)

        return train_cpm, val_cpm, test_cpm, train_ind, val_ind, test_ind


    def getdata(self, dataset, label=[], index=[], batch_size=128, train_size=0.9):

        self.batch_size = batch_size

        if len(index) > 0:
            self.index = index
        else:
            self.index = np.arange(0, dataset.shape[1])

        if len(label) > 0:
            train_ind, val_ind, test_ind = [], [], []
            for ll in np.unique(label):
                indx = np.where(label == ll)[0]
                tt_size = int(train_size * sum(label == ll))
                _, _, _, train_subind, val_subind, test_subind = self.data_gen(dataset[indx, :], tt_size)
                train_ind.append(indx[train_subind])
                val_ind.append(indx[val_subind])
                test_ind.append(indx[test_subind])

            train_ind = np.concatenate(train_ind)
            val_ind = np.concatenate(val_ind)
            test_ind = np.concatenate(test_ind)
            train_set = dataset[train_ind, :]
            val_set = dataset[val_ind, :]
            test_set = dataset[test_ind, :]
            self.n_class = len(np.unique(label))
        else:
            tt_size = int(train_size * dataset.shape[0])
            train_set, val_set, test_set, train_ind, val_ind, test_ind = self.data_gen(dataset, tt_size)

        train_set_torch = torch.FloatTensor(train_set)
        train_ind_torch = torch.FloatTensor(train_ind)
        train_data = TensorDataset(train_set_torch, train_ind_torch)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

        val_set_torch = torch.FloatTensor(val_set)
        val_ind_torch = torch.FloatTensor(val_ind)
        validation_data = TensorDataset(val_set_torch, val_ind_torch)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)

        test_set_torch = torch.FloatTensor(test_set)
        test_ind_torch = torch.FloatTensor(test_ind)
        test_data = TensorDataset(test_set_torch, test_ind_torch)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False, pin_memory=True)

        data_set_troch = torch.FloatTensor(dataset[:, self.index])
        all_ind_torch = torch.FloatTensor(range(dataset.shape[0]))
        all_data = TensorDataset(data_set_troch, all_ind_torch)
        alldata_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

        return alldata_loader, train_loader, validation_loader, test_loader


    def init_model(self, input_dim, fc_dim=100, lowD_dim=5, x_drop=0.2, lr=.001, variational=False, trained_model=''):
        """
        Initialized the deep mixture model and its optimizer.

        input args:
            fc_dim: dimension of the hidden layer.
            lowD_dim: dimension of the latent representation.
            x_drop: dropout probability at the first (input) layer.
            s_drop: dropout probability of the state variable.
            lr: the learning rate of the optimizer, here Adam.
            n_arm: int value that indicates number of arms.
            lam: coupling factor in the cpl-mixVAE model.
            tau: temperature of the softmax layers, usually equals to 1/n_categories (0 < tau <= 1).
            beta: regularizer for the KL divergence term.
            hard: a boolean variable, True uses one-hot method that is used in Gumbel-softmax, and False uses the Gumbel-softmax function.
            state_det: a boolean variable, False uses sampling.
            trained_model: the path of a pre-trained model, in case you wish to initialized the network with a pre-trained network.
            momentum: a hyperparameter for batch normalization that updates its running statistics.
        """
        self.lowD_dim = lowD_dim
        self.input_dim = input_dim
        self.fc_dim = fc_dim
        self.model = VAE(input_dim=input_dim, fc_dim=fc_dim, latent_dim=lowD_dim, p_drop=x_drop, varitioanl=variational)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if self.gpu:
            self.model = self.model.cuda(self.device)

        if len(trained_model) > 0:
            print('Load the pre-trained model')
            # if you wish to load another model for evaluation
            loaded_file = torch.load(trained_model, map_location='cpu')
            self.model.load_state_dict(loaded_file['model_state_dict'])
            self.optimizer.load_state_dict(loaded_file['optimizer_state_dict'])


    def load_model(self, trained_model):
        loaded_file = torch.load(trained_model, map_location='cpu')
        self.model.load_state_dict(loaded_file['model_state_dict'])

        self.current_time = time.strftime('%Y-%m-%d-%H-%M-%S')


    def run(self, train_loader, test_loader, validation_loader, alldata_loader, n_epoch):
        """
        run the training of the cpl-mixVAE with the pre-defined parameters/settings
        pcikle used for saving the file

        input args
            data_df: a data frame including 'cluster_id', 'cluster', and 'class_label'
            train_loader: train dataloader
            test_loader: test dataloader
            validation_set:
            n_epoch: number of training epoch, without pruning
            n_epoch: number of training epoch, with pruning
            min_con: minimum value of consensus among a pair of arms
            temp: temperature of sampling

        return
            data_file_id: the path of the output dictionary.
        """
        # define current_time
        self.current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

        # initialized saving arrays
        train_loss = np.zeros(n_epoch)
        validation_loss = np.zeros(n_epoch)

        print("Start training...")
        for epoch in range(n_epoch):
            train_loss_val = 0
            t0 = time.time()
            self.model.train()

            for batch_indx, (data, d_idx), in enumerate(train_loader):
                data = Variable(data)
                if self.gpu:
                    data = data.cuda(self.device)

                self.optimizer.zero_grad()
                x_recon, z = self.model(x=data)
                loss = self.model.loss(x_recon, data)
                loss.backward()
                self.optimizer.step()
                train_loss_val += loss.data.item()


            train_loss[epoch] = train_loss_val / (batch_indx + 1)

            print('====> Epoch:{}, Total Loss: {:.4f}, Elapsed Time:{:.2f}'.format(epoch, train_loss[epoch], time.time() - t0))

            # validation
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.
                for batch_indx, (data_val, d_idx), in enumerate(validation_loader):
                    d_idx = d_idx.to(int)
                    if self.gpu:
                        data_val = data_val.cuda(self.device)

                    x_recon, z = self.model(x=data_val)
                    loss = self.model.loss(x_recon, data_val)
                    val_loss += loss.data.item()

            validation_loss[epoch] = val_loss / (batch_indx + 1)
            # total_val_loss[epoch] = val_loss / (batch_indx + 1)
            print('====> Validation Loss: {:.4f}'.format(validation_loss[epoch]))

        if self.save and n_epoch > 0:
            trained_model = self.folder + '/model/cpl_mixVAE_model_before_pruning_' + self.current_time + '.pth'
            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)

            # plot the learning curve of the network
            fig, ax = plt.subplots()
            ax.plot(range(n_epoch), train_loss, label='Training')
            ax.plot(range(n_epoch), validation_loss, label='Validation')
            ax.set_xlabel('# epoch', fontsize=16)
            ax.set_ylabel('loss value', fontsize=16)
            ax.set_title('Learning curve of the cpl-mixVAE for |z|=' + str(self.lowD_dim))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.legend()
            ax.figure.savefig(self.folder + '/model/learning_curve_z_' + str(self.lowD_dim) + '_' + self.current_time + '.png')
            plt.close("all")


        max_len = len(alldata_loader.dataset)
        z_smp = np.zeros((max_len, self.lowD_dim))
        recon = np.zeros((max_len, self.input_dim))
        total_loss_val = []
        self.model.eval()
        with torch.no_grad():
            for i, (data, d_idx) in enumerate(alldata_loader):
                data = Variable(data)
                d_idx = d_idx.to(int)
                if self.gpu:
                    data = data.cuda(self.device)

                x_recon, z = self.model(data)
                loss = self.model.loss(x_recon, data)
                total_loss_val.append(loss.data.item())
                z_smp[i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = z.detach().cpu().numpy()
                recon[i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = x_recon.detach().cpu().numpy()

        # save data
        data_file_id = self.folder + '/model/data_' + self.current_time

        if self.save:
            self.save_file(data_file_id,
                           train_loss=train_loss,
                           validation_loss=validation_loss,
                           total_loss=total_loss_val,
                           z=z_smp,
                           x_recon=recon)

        return data_file_id


    def eval_model(self, data_mat, batch_size=1000):

        data_set_troch = torch.FloatTensor(data_mat)
        indx_set_troch = torch.FloatTensor(np.arange(data_mat.shape[0]))
        all_data = TensorDataset(data_set_troch, indx_set_troch)
        self.batch_size = batch_size

        data_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

        self.model.eval()
        max_len = len(data_loader.dataset)
        z_smp = np.zeros((max_len, self.lowD_dim))
        recon = np.zeros((max_len, self.input_dim))
        total_loss_val = []
        self.model.eval()
        with torch.no_grad():
            for i, (data, d_idx) in enumerate(data_loader):
                data = Variable(data)
                d_idx = d_idx.to(int)
                if self.gpu:
                    data = data.cuda(self.device)

                x_recon, z = self.model(data)
                loss = self.model.loss(x_recon, data)
                total_loss_val.append(loss.data.item())
                z_smp[i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = z.detach().cpu().numpy()
                recon[i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = x_recon.detach().cpu().numpy()

        # save data
        data_file_id = self.folder + '/model/model_eval' #_pruning_' + str(len(prune_indx))

        if self.save:
            self.save_file(data_file_id,
                           total_loss=total_loss_val,
                           z=z_smp,
                           x_recon=recon)

        return data_file_id


    def save_file(self, fname, **kwargs):
        """
        Save data as a .p file using pickle.

        input args
            fname: the path of the pre-trained network.
            kwarg: keyword arguments for input variables e.g., x=[], y=[], etc.
        """

        f = open(fname + '.p', "wb")
        data = {}
        for k, v in kwargs.items():
            data[k] = v
        pickle.dump(data, f)
        f.close()

    def load_file(self, fname):
        """
        load data .p file using pickle. Make sure to use the same version of
        pcikle used for saving the file

        input args
            fname: the path of the pre-trained network.

        return
            data: a dictionary including the save dataset
        """

        data = pickle.load(open(fname + '.p', "rb"))
        return data


