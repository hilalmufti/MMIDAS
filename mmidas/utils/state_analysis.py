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
from utils.nn_model import deepClassifier
from sklearn.metrics import accuracy_score
from numpy.linalg import norm



class state_analyzer:
    def __init__(self, saving_folder='', device=None, eps=1e-8, save_flag=True):

        self.eps = eps
        self.save = save_flag
        self.folder = saving_folder
        self.device= device

        if device is None:
            self.gpu = False
            print('using CPU ...')
        else:
            self.gpu = True
            torch.cuda.set_device(device)
            gpu_device = torch.device('cuda:' + str(device))
            print('using GPU ' + torch.cuda.get_device_name(torch.cuda.current_device()))

    def data_gen(self, dataset):

        test_size = dataset.shape[0] - self.train_size
        train_cpm, test_cpm, train_ind, test_ind = train_test_split(
            dataset[:, self.index], np.arange(dataset.shape[0]), train_size=self.train_size, test_size=test_size, random_state=0)

        train_cpm, val_cpm, train_ind, val_ind = train_test_split(train_cpm, train_ind, train_size=self.train_size - test_size, test_size=test_size, random_state=0)

        # train_cpm = train_cpm[:-test_size, :]
        # train_ind = train_ind[:-test_size]
        all_cpm = dataset[:, self.index]

        return train_cpm, val_cpm, test_cpm, train_ind, val_ind, test_ind, all_cpm


    def getdata(self, x, y, index=[], batch_size=128, train_size=0.9):

        self.batch_size = batch_size
        self.train_size = int(train_size * x.shape[0])

        if len(index) > 0:
            self.index = index
        else:
            self.index = np.arange(0, x.shape[1])

        train_set, val_set, test_set, train_ind, val_ind, test_ind, data_set = self.data_gen(dataset=x)

        train_set_torch = torch.FloatTensor(train_set)
        train_label_torch = torch.FloatTensor(y[train_ind])
        train_ind_torch = torch.FloatTensor(train_ind)
        train_data = TensorDataset(train_set_torch, train_label_torch, train_ind_torch)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

        val_set_torch = torch.FloatTensor(val_set)
        val_label_torch = torch.FloatTensor(y[val_ind])
        val_ind_torch = torch.FloatTensor(val_ind)
        validation_data = TensorDataset(val_set_torch, val_label_torch, val_ind_torch)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)

        test_set_torch = torch.FloatTensor(test_set)
        test_label_torch = torch.FloatTensor(y[test_ind])
        test_ind_torch = torch.FloatTensor(test_ind)
        test_data = TensorDataset(test_set_torch, test_label_torch, test_ind_torch)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False, pin_memory=True)

        data_set_troch = torch.FloatTensor(data_set)
        data_label_troch = torch.FloatTensor(y)
        all_ind_torch = torch.FloatTensor(range(len(data_set)))
        all_data = TensorDataset(data_set_troch, data_label_troch, all_ind_torch)
        alldata_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

        return alldata_loader, train_loader, validation_loader, test_loader


    def init_model(self, input_dim, output_dim, meta_label, x_drop=0.0, lr=.001, std=0, c=0, trained_model='', momentum=.01, binary=False):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.meta_label = meta_label
        self.model = deepClassifier(input_dim=self.input_dim, output_dim=output_dim, x_drop=x_drop,
                                    device=self.device, n_std=std, eps=self.eps, momentum=momentum, binary=binary)
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


    def run(self, data_in, data_out, roi_weight, train_indx, test_indx, batch_size, n_epoch, fold):

        # define current_time
        self.current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

        data_set_troch_in = torch.FloatTensor(data_in[train_indx, :])
        data_set_troch_out = torch.FloatTensor(data_out[train_indx])
        indx_set_troch = torch.FloatTensor(train_indx)
        train_data = TensorDataset(data_set_troch_in, data_set_troch_out, indx_set_troch)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

        data_set_troch_in = torch.FloatTensor(data_in[test_indx, :])
        data_set_troch_out = torch.FloatTensor(data_out[test_indx])
        indx_set_troch = torch.FloatTensor(test_indx)
        test_data = TensorDataset(data_set_troch_in, data_set_troch_out, indx_set_troch)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

        # initialized saving arrays
        train_loss = np.zeros(n_epoch)
        validation_loss = np.zeros(n_epoch)

        print("Start training...")
        for epoch in range(n_epoch):
            train_loss_val = 0
            t0 = time.time()
            self.model.train()

            for batch_indx, (state, metadata, d_indx), in enumerate(train_loader):
                state = Variable(state)
                weight = torch.FloatTensor(roi_weight[d_indx.to(int)])
                if self.gpu:
                    state = state.cuda(self.device)
                    metadata = metadata.cuda(self.device)
                    weight = weight.cuda(self.device)

                self.optimizer.zero_grad()
                y_score = self.model(state)
                loss = self.model.loss(y_score, metadata)
                loss.backward()
                self.optimizer.step()
                train_loss_val += loss.data.item()

            train_loss[epoch] = train_loss_val / (batch_indx + 1)
            print('====> Epoch:{}, Loss: {:.4f}, Elapsed Time:{:.2f}'.format(epoch, train_loss[epoch], time.time() - t0))

            # validation
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch_indx, (val_x, val_y, test_idx), in enumerate(test_loader):
                    # weight = torch.FloatTensor(roi_weight[test_idx.to(int)])
                    if self.gpu:
                        val_x = val_x.cuda(self.device)
                        val_y = val_y.cuda(self.device)
                        # weight = weight.cuda(self.device)

                    y_pred = self.model(val_x, eval=True)
                    loss = self.model.loss(y_pred, val_y)
                    val_loss += loss.data.item()

            validation_loss[epoch] = np.mean(val_loss / (batch_indx + 1))
            print('====> Validation Loss: {:.4f}'.format(validation_loss[epoch]))
            torch.cuda.empty_cache()

        if self.save and n_epoch > 0:
                trained_model = self.folder + '/deepClassifier_' + self.meta_label + '_fold_' + str(fold) + '_' + self.current_time + '.pth'
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)
                # plot the learning curve of the network
                fig, ax = plt.subplots()
                ax.plot(range(n_epoch), train_loss, label='training')
                ax.plot(range(n_epoch), validation_loss, label='validation')
                ax.legend()
                ax.set_xlabel('# epoch', fontsize=16)
                ax.set_ylabel('loss value', fontsize=16)
                ax.set_title('Learning curve of classifier')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.figure.savefig(self.folder + '/learning_curve_classifier_' + self.meta_label + '_' + self.current_time + '.png')
                plt.close("all")

        # Evaluate the trained model
        max_len = len(test_loader.dataset)
        s_index = np.zeros(max_len)
        predict_score = np.zeros((max_len, self.output_dim))
        if self.output_dim > 1:
            meta_data = np.zeros((max_len, self.output_dim))
        else:
            meta_data = np.zeros(max_len)

        total_loss_val = []
        self.model.eval()
        with torch.no_grad():
            for i, (x, y, x_idx) in enumerate(test_loader):
                if self.gpu:
                    x = x.cuda(self.device)
                    y = y.cuda(self.device)

                y_pred = self.model(x, eval=True)
                loss = self.model.loss(y_pred, y)
                total_loss_val.append(loss.data.item())
                s_index[i * batch_size:min((i + 1) * batch_size, max_len)] = x_idx.detach().cpu().numpy()
                # for d in range(self.output_dim):
                predict_score[i * batch_size:min((i + 1) * batch_size, max_len), :] = y_pred.detach().cpu().numpy()
                if self.output_dim > 1:
                    meta_data[i * batch_size:min((i + 1) * batch_size, max_len), :] = y.detach().cpu().numpy()
                else:
                    meta_data[i * batch_size:min((i + 1) * batch_size, max_len)] = y.detach().cpu().numpy()

            print(f'Total loss: {np.mean(total_loss_val)}')

        data_dict = dict()
        data_dict['sample_indx'] = s_index
        data_dict['metadata'] = meta_data
        data_dict['prediction'] = predict_score
        data_dict['loss'] = np.mean(total_loss_val)
        f = open(self.folder + '/summary_' + self.meta_label + '_fold_' + str(fold) + '.p', "wb")
        pickle.dump(data_dict, f)
        f.close()
        return data_dict

    def eval_prediction(self, data_in, data_out, anno, fold, batch_size=1000):

        data_set_troch_in = torch.FloatTensor(data_in)
        data_set_troch_out = torch.FloatTensor(data_out)
        indx_set_troch = torch.FloatTensor(np.arange(data_in.shape[0]))
        all_data = TensorDataset(data_set_troch_in, data_set_troch_out, indx_set_troch)
        self.batch_size = batch_size

        data_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
        max_len = len(data_loader.dataset)
        s_index = np.zeros(max_len)
        predict_score = np.zeros((max_len, self.output_dim))
        if self.output_dim > 1:
            meta_data = np.zeros((max_len, self.output_dim))
        else:
            meta_data = np.zeros(max_len)

        total_loss_val = []
        for i, (x, y, x_idx) in enumerate(data_loader):
            if self.gpu:
                x = x.cuda(self.device)
                y = y.cuda(self.device)

            y_pred = self.model(x, eval=True)
            loss = self.model.loss(y_pred, y)
            total_loss_val.append(loss.data.item())
            s_index[i * batch_size:min((i + 1) * batch_size, max_len)] = x_idx.to(int).detach().cpu().numpy()
            # for d in range(self.output_dim):
            predict_score[i * batch_size:min((i + 1) * batch_size, max_len), :] = y_pred.detach().cpu().numpy()
            if self.output_dim > 1:
                meta_data[i * batch_size:min((i + 1) * batch_size, max_len), :] = y.detach().cpu().numpy()
            else:
                meta_data[i * batch_size:min((i + 1) * batch_size, max_len)] = y.detach().cpu().numpy()

        print(f'Total loss: {np.mean(total_loss_val)}')

        # pred_roi = np.argmax(perdict_score, axis=1)
        # true_roi = np.argmax(meta_data, axis=1)
        # weight = [sum(true_roi == xx) / len(true_roi) for xx in true_roi]
        # acc = accuracy_score(true_roi, pred_roi, sample_weight=weight)
        # print('------------------')
        # print('ACC : ' + str(acc))
        # print('------------------')

        # fig = plt.figure(figsize=(12, 7), dpi=300)
        # m_size = 2
        # alp = .5
        # cosine = np.sum(meta_data * predict_score, axis=1) / (norm(meta_data, axis=1) * norm(predict_score, axis=1))
        # high_sim = np.where(cosine > 0.8)[0]

        s_index = s_index.astype(int)
        fig = plt.figure()
        m_size = 3
        alp = .5
        # axs = fig.add_subplot(1, 1, 1, projection='3d')
        # A = predict_score#[high_sim]
        # bregion = anno['roi'][s_index]#[high_sim]
        # br_color = anno['color'][s_index]#[high_sim]
        # for br in np.unique(bregion):
        #     br_idx = np.where(bregion == br)[0]
        #     axs.scatter(A[br_idx, 0], A[br_idx, 1], A[br_idx, 1], color=br_color[br_idx], s=m_size, alpha=alp, label=br)
        #
        # axs.set_xlabel('S_1')
        # axs.set_ylabel('S_2')
        # axs.set_zlabel('S_3')
        # axs.set_title('State variables - Inhibitory Cells')
        # axs.legend()
        # plt.tight_layout()
        # plt.savefig(self.folder + '/state_norm_ccf_fold' + str(fold) + '.png', dpi=600)
        # plt.close()
        #
        # if self.output_dim == 1:
        #     axs = fig.add_subplot(1, 1, 1)
        #     axs.scatter(perdict_score, color=anno['color'][s_index], s=m_size, alpha=alp)
        #     axs.scatter(meta_data, color='black', s=m_size+2)
        #     axs.set_xlabel('Flatten Brain CCF (x)')
        if self.output_dim == 2:
            axs = fig.add_subplot(1, 1, 1)
            axs.scatter(predict_score[:, 0], predict_score[:, 1], color=anno['color'][s_index], s=m_size, alpha=alp)
            axs.scatter(meta_data[:, 0], meta_data[:, 1], color='black', s=m_size+2)
            axs.set_xlabel('Flatten Brain CCF (x)')
            axs.set_ylabel('Flatten Brain CCF (y)')
        elif self.output_dim == 3:
            axs = fig.add_subplot(1, 1, 1, projection='3d')
            axs.scatter(predict_score[:, 0], predict_score[:, 1], predict_score[:, 2], color=anno['color'][s_index], s=m_size, alpha=alp)
            # axs.scatter(meta_data[:, 0], meta_data[:, 1], meta_data[:, 2], color='black', s=m_size + 2)
            axs.set_xlabel('Brain CCF (x)')
            axs.set_ylabel('Brain CCF (y)')
            axs.set_zlabel('Brain CCF (z)')

        plt.tight_layout()
        plt.savefig(self.folder + '/state_ccf_fold_' + str(fold) + '.png', dpi=600)
        #
        # categories = anno['cluster'][s_index]
        # meta_pred_c = []
        # meta_ref_c = []
        # uniq_cat = np.unique(categories)
        # for ic, cc in enumerate(uniq_cat):
        #     indx = np.where(categories == cc)[0]
        #     meta_pred_c.append(np.median(perdict_score[indx], axis=0))
        #     meta_ref_c.append(np.median(meta_data[indx], axis=0))
        #
        # meta_pred_c = np.array(meta_pred_c)
        # meta_ref_c = np.array(meta_ref_c)
        # plt.figure()
        # plt.plot(np.arange(len(uniq_cat)), meta_ref_c[:, 0], label='X - Given')
        # plt.plot(np.arange(len(uniq_cat)), meta_pred_c[:, 0], label='X - Prediction')
        # plt.plot(np.arange(len(uniq_cat)), meta_ref_c[:, 1], label='Y - Given')
        # plt.plot(np.arange(len(uniq_cat)), meta_pred_c[:, 1], label='Y - Prediction')
        # plt.plot(np.arange(len(uniq_cat)), meta_ref_c[:, 2], label='Z - Given')
        # plt.plot(np.arange(len(uniq_cat)), meta_pred_c[:, 2], label='Z - Prediction')
        # plt.xticks(np.arange(len(uniq_cat)), uniq_cat, rotation='vertical')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(self.folder + '/state_cct_perC_fold_' + str(fold) + '.png', dpi=600)
        #
        # rois = anno['roi'][s_index]
        # meta_pred_roi = []
        # meta_ref_roi = []
        # uniq_roi = np.unique(rois)
        # for ir, rr in enumerate(uniq_roi):
        #     indx = np.where(rois == rr)[0]
        #     meta_pred_roi.append(np.median(perdict_score[indx], axis=0))
        #     meta_ref_roi.append(np.median(meta_data[indx], axis=0))
        #
        # meta_pred_roi = np.array(meta_pred_roi)
        # meta_ref_roi = np.array(meta_ref_roi)
        # plt.figure()
        # plt.plot(np.arange(len(uniq_roi)), meta_ref_roi[:, 0], label='X - Given')
        # plt.plot(np.arange(len(uniq_roi)), meta_pred_roi[:, 0], label='X - Prediction')
        # plt.plot(np.arange(len(uniq_roi)), meta_ref_roi[:, 1], label='Y - Given')
        # plt.plot(np.arange(len(uniq_roi)), meta_pred_roi[:, 1], label='Y - Prediction')
        # plt.plot(np.arange(len(uniq_roi)), meta_ref_roi[:, 2], label='Z - Given')
        # plt.plot(np.arange(len(uniq_roi)), meta_pred_roi[:, 2], label='Z - Prediction')
        # plt.xticks(np.arange(len(uniq_roi)), uniq_roi, rotation='vertical')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(self.folder + '/state_cct_perROI_fold' + str(fold) + '.png', dpi=600)
        # plt.close('all')

        data_dict = dict()
        data_dict['sample_indx'] = s_index
        data_dict['metadata'] = meta_data
        data_dict['prediction'] = predict_score
        # data_dict['acc'] = acc
        # data_dict['meta_pred_c'] = meta_pred_c
        # data_dict['meta_ref_c'] = meta_ref_c
        # data_dict['meta_pred_roi'] = meta_pred_roi
        # data_dict['meta_ref_roi'] = meta_ref_roi

        return data_dict


