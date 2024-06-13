import pickle
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
import time
import torch
from torch.autograd import Variable
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from .augmentation.udagan import *
from .nn_model import mixVAE_model, loss_fn

# fsdp imports
import os
import subprocess 
import random
from pprint import pprint
from functools import partial
from operator import getitem
from contextlib import contextmanager
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR 
import torch.distributed as dist 
import torch.multiprocessing as mp 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import builtins as bis
from typing import List, Type

LABEL_ALIGN = '<'
LABEL_WIDTH = 30
VALUE_WIDTH = 30
VALUE_ALIGN = '<'
PRECISION = 4
EPOCH_WIDTH = LABEL_WIDTH + VALUE_WIDTH

# utils 

def mems():
    cmd = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'])
    return [int(x) for x in cmd.decode('utf-8').strip().split('\n')]

def identity(x):
    return x

def argmax(xs, *, key=identity):
    return max(enumerate(xs), key=lambda x: key(x[1]))[0]

def max_mem():
  return argmax(mems())

def dev_t(x, t=int):
  match t:
    case bis.int:
      return x 
    case bis.str:
      return f'cuda:{x}'
    case torch.device:
      return torch.device(f'cuda:{x}')
    case _:
      raise ValueError(f'dev_t: {t}')

def to(x, dst):
  return x.to(dst)

def bsize(x):
  return x.batch_size

def item(tensor): 
  return tensor.item()

def join(s, *args, **kwargs): 
    return s.join(*args, **kwargs)
      

def _label(s, align=LABEL_ALIGN, width=LABEL_WIDTH):
    return f"{s:{align}{width}}"
    
def _value(s, align=VALUE_ALIGN, width=VALUE_WIDTH, precision=PRECISION):
    return f"{s:{align}{width}.{precision}f}"

def set_seed(seed):
  print(f"warning: setting seed {seed}")
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

class cpl_mixVAE:
    def __init__(self, saving_folder='', aug_file='', device=None, eps=1e-8, save_flag=True, seed=546):
        """
        Initialized the cpl_mixVAE class.

        input args:
            saving_folder: a string that indicates the folder to save the model(s) and file(s).
            aug_file: a string that indicates the file of the pre-trained augmenter.
            device: computing device, either 'cpu' or 'cuda'.
            eps: a small constant value to fix computation overflow.
            save_flag: a boolean variable, if True, the model is saved.
        """
        set_seed(seed)

        self.eps = eps
        self.save = save_flag
        self.folder = saving_folder
        self.aug_file = aug_file
        self.device = device

        if device is None:
            self.device = torch.device('cpu')
            print('---> Computional node is not assigned, using CPU!')
        else:
            if device == 'cpu':
                self.device = torch.device('cpu')
                print('---> Using CPU!')
            else:
                self.device = dev_t(max_mem(), torch.device)
                torch.cuda.set_device(self.device)
                print('device: ' + torch.cuda.get_device_name(torch.cuda.current_device()))

        if self.aug_file:
            self.aug_model = torch.load(self.aug_file)
            self.aug_param = self.aug_model['parameters']
            self.netA = Augmenter(noise_dim=self.aug_param['num_n'],
                                latent_dim=self.aug_param['num_z'],
                                n_zim=self.aug_param['n_zim'],
                                input_dim=self.aug_param['n_features'])
            # Load the trained augmenter weights
            self.netA.load_state_dict(self.aug_model['netA'])
            self.netA = self.netA.to(self.device)


    def init_model(self, categories, state_dim, input_dim, fc_dim=100, lowD_dim=10, x_drop=0.5, s_drop=0.2, lr=.001,
                   lam=1, lam_pc=1, arms=2, temp=1., tau=0.005, beta=1., hard=False, variational=True, ref_prior=False,
                   trained_model='', n_pr=0, momentum=.01, mode='MSE', n_arm=-1, n_categories=-1):
        if n_arm != -1:
            raise ValueError("n_arm is deprecated, use arms instead")
        if n_categories != -1:
            raise ValueError("n_categories is deprecated, \
                              use categories instead")
        """
        Initialized the deep mixture model and its optimizer.

        input args:
            categories: number of categories of the latent variables.
            state_dim: dimension of the state variable.
            input_dim: input dimension (size of the input layer).
            fc_dim: dimension of the hidden layer.
            lowD_dim: dimension of the latent representation.
            x_drop: dropout probability at the first (input) layer.
            s_drop: dropout probability of the state variable.
            lr: the learning rate of the optimizer, here Adam.
            lam: coupling factor in the cpl-mixVAE model.
            lam_pc: coupling factor for the prior categorical variable.
            arms: int value that indicates number of arms.
            temp: temperature of sampling
            tau: temperature of the softmax layers, usually equals to 1/categories (0 < tau <= 1).
            beta: regularizer for the KL divergence term.
            hard: a boolean variable, True uses one-hot method that is used in Gumbel-softmax, and False uses the Gumbel-softmax function.
            variational: a boolean variable for variational mode, False mode does not use sampling.
            ref_prior: a boolean variable, True uses the reference prior for the categorical variable.
            trained_model: a pre-trained model, in case you want to initialized the network with a pre-trained network.
            n_pr: number of pruned categories, only if you want to initialize the network with a pre-trained network.
            momentum: a hyperparameter for batch normalization that updates its running statistics.
            mode: the loss function, either 'MSE' or 'ZINB'.
        """
        print("warning: init_model is deprecated, use init instead")
        self.lowD_dim = lowD_dim
        self.categories = categories
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.temp = temp
        self.arms = arms
        self.fc_dim = fc_dim
        self.ref_prior = ref_prior
        self.model = mixVAE_model(input_dim=self.input_dim, fc_dim=fc_dim, n_categories=self.categories, state_dim=self.state_dim,
                                lowD_dim=lowD_dim, x_drop=x_drop, s_drop=s_drop, n_arm=self.arms, lam=lam, lam_pc=lam_pc,
                                tau=tau, beta=beta, hard=hard, variational=variational, device=self.device, eps=self.eps,
                                ref_prior=ref_prior, momentum=momentum, loss_mode=mode)
        
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if len(trained_model) > 0:
            print('Load the pre-trained model')
            # if you wish to load another model for evaluation
            loaded_file = torch.load(trained_model, map_location='cpu')
            self.model.load_state_dict(loaded_file['model_state_dict'])
            self.optimizer.load_state_dict(loaded_file['optimizer_state_dict'])
            self.is_init = False
            self.n_pr = n_pr
        else:
            self.is_init = True
            self.n_pr = 0

    def init(self, categories, state_dim, input_dim, fc_dim=100, lowD_dim=10, x_drop=0.5, s_drop=0.2, lr=.001,
                   lam=1, lam_pc=1, arms=2, temp=1., tau=0.005, beta=1., hard=False, variational=True, ref_prior=False,
                   trained_model='', n_pr=0, momentum=.01, mode='MSE', n_arm=-1, n_categories=-1):
        if n_arm != -1:
            raise ValueError("n_arm is deprecated, use arms instead")
        if n_categories != -1:
            raise ValueError("n_categories is deprecated, \
                              use categories instead")
        """
        Initialized the deep mixture model and its optimizer.

        input args:
            categories: number of categories of the latent variables.
            state_dim: dimension of the state variable.
            input_dim: input dimension (size of the input layer).
            fc_dim: dimension of the hidden layer.
            lowD_dim: dimension of the latent representation.
            x_drop: dropout probability at the first (input) layer.
            s_drop: dropout probability of the state variable.
            lr: the learning rate of the optimizer, here Adam.
            lam: coupling factor in the cpl-mixVAE model.
            lam_pc: coupling factor for the prior categorical variable.
            arms: int value that indicates number of arms.
            temp: temperature of sampling
            tau: temperature of the softmax layers, usually equals to 1/categories (0 < tau <= 1).
            beta: regularizer for the KL divergence term.
            hard: a boolean variable, True uses one-hot method that is used in Gumbel-softmax, and False uses the Gumbel-softmax function.
            variational: a boolean variable for variational mode, False mode does not use sampling.
            ref_prior: a boolean variable, True uses the reference prior for the categorical variable.
            trained_model: a pre-trained model, in case you want to initialized the network with a pre-trained network.
            n_pr: number of pruned categories, only if you want to initialize the network with a pre-trained network.
            momentum: a hyperparameter for batch normalization that updates its running statistics.
            mode: the loss function, either 'MSE' or 'ZINB'.
        """
        self.lowD_dim = lowD_dim
        self.categories = categories
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.temp = temp
        self.arms = arms
        self.fc_dim = fc_dim
        self.ref_prior = ref_prior
        self.mode = mode
        self.lam_pc = lam_pc
        self.lam = lam
        self.is_var = variational
        self.beta = beta
        self.model = mixVAE_model(input_dim=self.input_dim, fc_dim=fc_dim, n_categories=self.categories, state_dim=self.state_dim,
                                lowD_dim=lowD_dim, x_drop=x_drop, s_drop=s_drop, n_arm=self.arms, lam=lam, lam_pc=lam_pc,
                                tau=tau, beta=beta, hard=hard, variational=variational, device=self.device, eps=self.eps,
                                ref_prior=ref_prior, momentum=momentum, loss_mode=mode)
        
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if len(trained_model) > 0:
            print('Load the pre-trained model')
            # if you wish to load another model for evaluation
            loaded_file = torch.load(trained_model, map_location='cpu')
            self.model.load_state_dict(loaded_file['model_state_dict'])
            self.optimizer.load_state_dict(loaded_file['optimizer_state_dict'])
            self.is_init = False
            self.n_pr = n_pr
        else:
            self.is_init = True
            self.n_pr = 0


    def load_model(self, trained_model):
        loaded_file = torch.load(trained_model, map_location='cpu')
        self.model.load_state_dict(loaded_file['model_state_dict'])

        self.current_time = time.strftime('%Y-%m-%d-%H-%M-%S')


    # TODO
    # [] Make this not horrific
    def train(self, train_loader, test_loader, n_epoch, n_epoch_p, c_p=0, c_onehot=0, min_con=.5, max_prun_it=0):
        """
        run the training of the cpl-mixVAE with the pre-defined parameters/settings
        pcikle used for saving the file

        input args
            train_loader: train dataloader.
            test_loader: test dataloader.
            n_epoch: number of training epoch, without pruning.
            n_epoch_p: number of training epoch, with pruning.
            c_p: the prior categorical variable, only if ref_prior is True.
            c_onehot: the one-hot representation of the prior categorical variable, only if ref_prior is True.
            min_con: minimum value of consensus among pair of arms.
            max_prun_it: maximum number of pruning iterations.
            mode: the loss function, either 'MSE' or 'ZINB'.

        return
            data_file_id: the output dictionary.
        """

        # define current_time
        self.current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

        # initialized saving arrays
        train_loss = np.zeros(n_epoch)
        validation_loss = np.zeros(n_epoch)
        train_loss_joint = np.zeros(n_epoch)
        train_entropy = np.zeros(n_epoch)
        train_distance = np.zeros(n_epoch)
        train_minVar = np.zeros(n_epoch)
        train_log_distance = np.zeros(n_epoch)
        train_recon = np.zeros((self.arms, n_epoch))
        train_loss_KL = np.zeros((self.arms, self.categories, n_epoch))
        validation_rec_loss = np.zeros(n_epoch)
        bias_mask = torch.ones(self.categories)
        weight_mask = torch.ones((self.categories, self.lowD_dim))
        fc_mu = torch.ones((self.state_dim, self.categories + self.lowD_dim))
        fc_sigma = torch.ones((self.state_dim, self.categories + self.lowD_dim))
        f6_mask = torch.ones((self.lowD_dim, self.state_dim + self.categories))

        bias_mask = bias_mask.to(self.device)
        weight_mask = weight_mask.to(self.device)
        fc_mu = fc_mu.to(self.device)
        fc_sigma = fc_sigma.to(self.device)
        f6_mask = f6_mask.to(self.device)
        batch_size = train_loader.batch_size

        if self.is_init:
            print("Start training ...")
            pbar = tqdm(range(n_epoch))
            for epoch in pbar:
                train_loss_val = 0
                train_jointloss_val = 0
                train_dqc = 0
                log_dqc = 0
                entr = 0
                var_min = 0
                t0 = time.time()
                train_loss_rec = np.zeros(self.arms)
                train_KLD_cont = np.zeros((self.arms, self.categories))
                self.model.train()

                for batch_indx, (data, d_idx), in enumerate(train_loader):
                    data = Variable(data).to(self.device)
                    d_idx = d_idx.to(int)
                        
                    trans_data = []
                    tt = time.time()
                    for arm in range(self.arms):
                        if self.aug_file:
                            noise = torch.randn(batch_size, self.aug_param['num_n'], device=self.device)
                            _, gen_data = self.netA(data, noise, True, self.device)
                            if self.aug_param['n_zim'] > 1:
                                data_bin = 0. * data
                                data_bin[data > self.eps] = 1.
                                fake_data = gen_data[:, :self.aug_param['n_features']] * data_bin
                                trans_data.append(fake_data)
                            else:
                                trans_data.append(gen_data)
                        else:
                            trans_data.append(data)

                    if self.ref_prior:
                        c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                        prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                    else:
                        c_bin = 0.
                        prior_c = 0.

                    self.optimizer.zero_grad()
                    recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, log_qc = self.model(x=trans_data, temp=self.temp, prior_c=prior_c)
                    loss, loss_rec, loss_joint, entropy, dist_c, d_qc, KLD_cont, min_var_0, loglikelihood = \
                        self.model.loss(recon_batch, p_x, r_x, trans_data, mu, log_var, qc, c, c_bin)
                    loss.backward()
                    self.optimizer.step()
                    train_loss_val += loss.data.item()
                    train_jointloss_val += loss_joint
                    train_dqc += d_qc
                    log_dqc += dist_c
                    entr += entropy
                    var_min += min_var_0.data.item()

                    for arm in range(self.arms):
                        train_loss_rec[arm] += loss_rec[arm].data.item() / self.input_dim

                train_loss[epoch] = train_loss_val / (batch_indx + 1)
                train_loss_joint[epoch] = train_jointloss_val / (batch_indx + 1)
                train_distance[epoch] = train_dqc / (batch_indx + 1)
                train_entropy[epoch] = entr / (batch_indx + 1)
                train_log_distance[epoch] = log_dqc / (batch_indx + 1)
                train_minVar[epoch] = var_min / (batch_indx + 1)

                for arm in range(self.arms):
                    train_recon[arm, epoch] = train_loss_rec[arm] / (batch_indx + 1)
                    for cc in range(self.categories):
                        train_loss_KL[arm, cc, epoch] = train_KLD_cont[arm, cc] / (batch_indx + 1)

                # print('====> Epoch:{}, Total Loss: {:.4f}, Rec_arm_1: {'':.4f}, Rec_arm_2: {'':.4f}, Joint Loss: {:.4f}, '
                #       'Entropy: {:.4f}, Distance: {:.4f}, Elapsed Time:{:.2f}'.format(
                #     epoch, train_loss[epoch], train_recon[0, epoch], train_recon[1, epoch], train_loss_joint[epoch],
                #     train_entropy[epoch], train_distance[epoch], time.time() - t0))
                
                # validation
                self.model.eval()
                with torch.no_grad():
                    val_loss_rec = 0.
                    val_loss = 0.
                    if test_loader.batch_size > 1:
                        for batch_indx, (data_val, d_idx), in enumerate(test_loader):
                            data_val = data_val.to(self.device)
                            d_idx = d_idx.to(int)
                            trans_val_data = []
                            for arm in range(self.arms):
                               trans_val_data.append(data_val)

                            if self.ref_prior:
                                c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                                prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                            else:
                                c_bin = 0.
                                prior_c = 0.

                            recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, prior_c=prior_c, eval=True)
                            loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data, mu, log_var, qc, c, c_bin)
                            val_loss += loss.data.item()
                            for arm in range(self.arms):
                                val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                    else:
                        batch_indx = 0
                        data_val, d_idx = test_loader.dataset.tensors
                        data_val = data_val.to(self.device)
                        d_idx = d_idx.to(int)
                        trans_val_data = []
                        for arm in range(self.arms):
                            trans_val_data.append(data_val)

                        if self.ref_prior:
                            c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                            prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                        else:
                            c_bin = 0.
                            prior_c = 0.

                        recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data,
                                                                                            temp=self.temp,
                                                                                            prior_c=prior_c, eval=True)
                        loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x,
                                                                                       trans_val_data, mu, log_var, qc,
                                                                                       c, c_bin)
                        val_loss = loss.data.item()
                        for arm in range(self.arms):
                            val_loss_rec += loss_rec[arm].data.item() / self.input_dim

                validation_rec_loss[epoch] = val_loss_rec / (batch_indx + 1) / self.arms
                validation_loss[epoch] = val_loss / (batch_indx + 1)

                # wandb.log({
                #     'train_loss': train_loss[epoch],
                #     'train_loss_joint': train_loss_joint[epoch],
                #     'train_entropy': train_entropy[epoch],
                #     'train_distance': train_distance[epoch],
                #     'train_minVar': train_minVar[epoch],
                #     'train_log_distance': train_log_distance[epoch],
                #     'train_recon_0': train_recon[0, epoch],
                #     'train_recon_1': train_recon[1, epoch],
                #     'train_loss_KL_0': train_loss_KL[0, :, epoch],
                #     'train_loss_KL_1': train_loss_KL[1, :, epoch],
                #     'validation_loss': validation_loss[epoch],
                #     'validation_rec_loss': validation_rec_loss[epoch]
                # })
                # print(f"{'-- Epoch: ' + str(epoch) + ' --':^{EPOCH_WIDTH}}")")
                print(f" -- EPOCH {str(epoch)} -- ".center((EPOCH_WIDTH // 2) + 5, ' '))


                print(_label('Total Loss:'), _value(train_loss[epoch]))
                print(_label('Rec_arm_1:'), _value(train_recon[0, epoch]))
                print(_label('Rec_arm_2:'), _value(train_recon[1, epoch]))
                print(_label('Joint Loss:'), _value(train_loss_joint[epoch]))
                print(_label('Entropy:'), _value(train_entropy[epoch]))
                print(_label('Distance:'), _value(train_distance[epoch]))
                print(_label('Elapsed Time:'), _value(time.time() - t0))
                print(_label('Validation Loss:'), _value(validation_loss[epoch]))
                print(_label('Validation Rec. Loss:'), _value(validation_rec_loss[epoch]))

                pbar.set_postfix({'Total Loss': train_loss[epoch],
                                  'Rec_arm_1': train_recon[0, epoch],
                                  'Rec_arm_2': train_recon[1, epoch],
                                  'Joint Loss': train_loss_joint[epoch],
                                  'Entropy': train_entropy[epoch],
                                  'Distance': train_distance[epoch], 
                                  'Elapsed Time': time.time() - t0, 
                                  'Validation Loss': validation_loss[epoch], 
                                  'Validation Rec. Loss': 
                                    validation_rec_loss[epoch]})

                # print('====> Validation Total Loss: {:.4f}, Rec. Loss: {:.4f}'.format(validation_loss[epoch], validation_rec_loss[epoch]))

        # ------------------- Pruning -------------------


            if self.save and n_epoch > 0:
                trained_model = self.folder + '/model/cpl_mixVAE_model_before_pruning_' + self.current_time + '.pth'
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)
                bias = self.model.fcc[0].bias.detach().cpu().numpy()
                mask = range(len(bias))
                prune_indx = []
                # plot the learning curve of the network
                fig, ax = plt.subplots()
                ax.plot(range(n_epoch), train_loss, label='Training')
                ax.plot(range(n_epoch), validation_loss, label='Validation')
                ax.set_xlabel('# epoch', fontsize=16)
                ax.set_ylabel('loss value', fontsize=16)
                ax.set_title('Learning curve of the cpl-mixVAE for K=' + str(self.categories) + ' and S=' + str(self.state_dim))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.legend()
                ax.figure.savefig(self.folder + '/model/learning_curve_before_pruning_K_' + str(self.categories) + '_' + self.current_time + '.png')
                plt.close("all")

        if n_epoch_p > 0:
            # initialized pruning parameters of the layer of the discrete variable
            bias = self.model.fcc[0].bias.detach().cpu().numpy()
            pruning_mask = np.where(bias != 0.)[0]
            prune_indx = np.where(bias == 0.)[0]
            stop_prune = False
        else:
            stop_prune = True

        pr = self.n_pr
        ind = []
        while not stop_prune:
            predicted_label = np.zeros((self.arms, len(train_loader.dataset)))

            # Assessment over all dataset
            self.model.eval()
            with torch.no_grad():
                for i, (data, d_idx) in enumerate(train_loader):
                    data = data.to(self.device)
                    d_idx = d_idx.to(int)
                    trans_data = []
                    for arm in range(self.arms):
                        trans_data.append(data)

                    if self.ref_prior:
                        c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                        prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                    else:
                        c_bin = 0.
                        prior_c = 0.

                    recon, p_x, r_x, x_low, z_category, state, z_smp, mu, log_sigma, _ = self.model(trans_data, self.temp, prior_c, mask=pruning_mask, eval=True)

                    for arm in range(self.arms):
                        z_encoder = z_category[arm].cpu().data.view(z_category[arm].size()[0], self.categories).detach().numpy()
                        predicted_label[arm, i * batch_size:min((i + 1) * batch_size, len(train_loader.dataset))] = np.argmax(z_encoder, axis=1)

            c_agreement = []
            for arm_a in range(self.arms):
                pred_a = predicted_label[arm_a, :]
                for arm_b in range(arm_a + 1, self.arms):
                    pred_b = predicted_label[arm_b, :]
                    armA_vs_armB = np.zeros((self.categories, self.categories))

                    for samp in range(pred_a.shape[0]):
                        armA_vs_armB[pred_a[samp].astype(int), pred_b[samp].astype(int)] += 1

                    num_samp_arm = []
                    for ij in range(self.categories):
                        sum_row = armA_vs_armB[ij, :].sum()
                        sum_column = armA_vs_armB[:, ij].sum()
                        num_samp_arm.append(max(sum_row, sum_column))

                    armA_vs_armB = np.divide(armA_vs_armB, np.array(num_samp_arm), out=np.zeros_like(armA_vs_armB),
                                             where=np.array(num_samp_arm) != 0)
                    c_agreement.append(np.diag(armA_vs_armB))
                    ind_sort = np.argsort(c_agreement[-1])
                    plt.figure()
                    plt.imshow(armA_vs_armB[:, ind_sort[::-1]][ind_sort[::-1]], cmap='binary')
                    plt.colorbar()
                    plt.xlabel('arm_' + str(arm_a), fontsize=20)
                    plt.xticks(range(self.categories), range(self.categories))
                    plt.yticks(range(self.categories), range(self.categories))
                    plt.ylabel('arm_' + str(arm_b), fontsize=20)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('|c|=' + str(self.categories), fontsize=20)
                    plt.savefig(self.folder + '/consensus_' + str(pr) + '_arm_' + str(arm_a) + '_arm_' + str(arm_b) + '.png', dpi=600)
                    plt.close("all")

            c_agreement = np.mean(c_agreement, axis=0)
            agreement = c_agreement[pruning_mask]
            if (np.min(agreement) <= min_con) and pr < max_prun_it:
                if pr > 0:
                    ind_min = pruning_mask[np.argmin(agreement)]
                    ind_min = np.array([ind_min])
                    ind = np.concatenate((ind, ind_min))
                else:
                    ind_min = pruning_mask[np.argmin(agreement)]
                    if len(prune_indx) > 0:
                        ind_min = np.array([ind_min])
                        ind = np.concatenate((prune_indx, ind_min))
                    else:
                        ind.append(ind_min)
                    ind = np.array(ind)

                ind = ind.astype(int)
                bias_mask[ind] = 0.
                weight_mask[ind, :] = 0.
                fc_mu[:, self.lowD_dim + ind] = 0.
                fc_sigma[:, self.lowD_dim + ind] = 0.
                f6_mask[:, ind] = 0.
                stop_prune = False
            else:
                print('No more pruning!')
                stop_prune = True

            if not stop_prune:
                print("Continue training with pruning ...")
                print(f"Pruned categories: {ind}")
                bias = bias_mask.detach().cpu().numpy()
                pruning_mask = np.where(bias != 0.)[0]
                train_loss = np.zeros(n_epoch_p)
                validation_rec_loss = np.zeros(n_epoch_p)
                total_val_loss = np.zeros(n_epoch_p)
                train_loss_joint = np.zeros(n_epoch_p)
                train_entropy = np.zeros(n_epoch_p)
                train_distance = np.zeros(n_epoch_p)
                train_minVar = np.zeros(n_epoch_p)
                train_log_distance = np.zeros(n_epoch_p)
                train_recon = np.zeros((self.arms, n_epoch_p))
                train_loss_KL = np.zeros((self.arms, self.categories, n_epoch_p))

                for arm in range(self.arms):
                    prune.custom_from_mask(self.model.fcc[arm], 'weight', mask=weight_mask)
                    prune.custom_from_mask(self.model.fcc[arm], 'bias', mask=bias_mask)
                    prune.custom_from_mask(self.model.fc_mu[arm], 'weight', mask=fc_mu)
                    prune.custom_from_mask(self.model.fc_sigma[arm], 'weight', mask=fc_sigma)
                    prune.custom_from_mask(self.model.fc6[arm], 'weight', mask=f6_mask)

                pbar = tqdm(range(n_epoch_p))
                for epoch in pbar:
                    # training
                    train_loss_val = 0
                    train_jointloss_val = 0
                    train_dqz = 0
                    log_dqz = 0
                    entr = 0
                    var_min = 0
                    t0 = time.time()
                    train_loss_rec = np.zeros(self.arms)
                    train_KLD_cont = np.zeros((self.arms, self.categories))
                    ti = np.zeros(len(train_loader))
                    self.model.train()
                    # training
                    for batch_indx, (data, d_idx), in enumerate(train_loader):
                        # for data in train_loader:
                        data = Variable(data).to(self.device)
                        d_idx = d_idx.to(int)
                        data_bin = 0. * data
                        data_bin[data > 0.] = 1.
                        trans_data = []
                        origin_data = []
                        trans_data.append(data)
                        tt = time.time()
                        w_param, bias_param, activ_param = 0, 0, 0
                        for arm in range(self.arms-1):
                            if self.aug_file:
                                noise = torch.randn(batch_size, self.aug_param['num_n']).to(self.device)
                                _, gen_data = self.netA(data, noise, True, self.device)
                                if self.aug_param['n_zim'] > 1:
                                    data_bin = 0. * data
                                    data_bin[data > self.eps] = 1.
                                    fake_data = gen_data[:, :self.aug_param['n_features']] * data_bin
                                    trans_data.append(fake_data)
                                else:
                                    trans_data.append(gen_data)
                            else:
                                trans_data.append(data)

                        if self.ref_prior:
                            c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                            prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                        else:
                            c_bin = 0.
                            prior_c = 0.

                        self.optimizer.zero_grad()
                        recon_batch, p_x, r_x, x_low, qz, s, z, mu, log_var, log_qz = self.model(trans_data, self.temp, prior_c, mask=pruning_mask)
                        loss, loss_rec, loss_joint, entropy, dist_z, d_qz, KLD_cont, min_var_0, _ = self.model.loss(recon_batch, p_x, r_x,
                                                                                        trans_data, mu, log_var, qz, z, c_bin)

                        loss.backward()
                        self.optimizer.step()
                        ti[batch_indx] = time.time() - tt
                        train_loss_val += loss.data.item()
                        train_jointloss_val += loss_joint
                        train_dqz += d_qz
                        log_dqz += dist_z
                        entr += entropy
                        var_min += min_var_0.data.item()

                        for arm in range(self.arms):
                            train_loss_rec[arm] += loss_rec[arm].data.item() / self.input_dim

                    train_loss[epoch] = train_loss_val / (batch_indx + 1)
                    train_loss_joint[epoch] = train_jointloss_val / (batch_indx + 1)
                    train_distance[epoch] = train_dqz / (batch_indx + 1)
                    train_entropy[epoch] = entr / (batch_indx + 1)
                    train_log_distance[epoch] = log_dqz / (batch_indx + 1)
                    train_minVar[epoch] = var_min / (batch_indx + 1)

                    for arm in range(self.arms):
                        train_recon[arm, epoch] = train_loss_rec[arm] / (batch_indx + 1)
                        for c in range(self.categories):
                            train_loss_KL[arm, c, epoch] = train_KLD_cont[arm, c] / (batch_indx + 1)

                    

                    # print('====> Epoch:{}, Total Loss: {:.4f}, Rec_arm_1: {'
                    #       ':.4f}, Rec_arm_2: {:.4f}, Joint Loss: {:.4f}, Entropy: {:.4f}, Distance: {:.4f}, Elapsed Time:{:.2f}'.format(
                    #     epoch, train_loss[epoch], train_recon[0, epoch], train_recon[1, epoch], train_loss_joint[epoch],
                    #     train_entropy[epoch], train_distance[epoch], time.time() - t0))

                    # validation
                    self.model.eval()
                    with torch.no_grad():
                        val_loss_rec = 0.
                        val_loss = 0.
                        if test_loader.batch_size > 1:
                            for batch_indx, (data_val, d_idx), in enumerate(test_loader):
                                d_idx = d_idx.to(int)
                                data_val = data_val.to(self.device)
                                    
                                trans_val_data = []
                                for arm in range(self.arms):
                                    trans_val_data.append(data_val)

                                if self.ref_prior:
                                    c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                                    prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                                else:
                                    c_bin = 0.
                                    prior_c = 0.

                                recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, prior_c=prior_c,
                                                                                        eval=True, mask=pruning_mask)
                                loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data,
                                                                                            mu, log_var, qc, c, c_bin)
                                val_loss += loss.data.item()
                                for arm in range(self.arms):
                                    val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                        else:
                            batch_indx = 0
                            data_val, d_idx = test_loader.dataset.tensors
                            data_val = data_val.to(self.device)
                            d_idx = d_idx.to(int)
                            trans_val_data = []
                            for arm in range(self.arms):
                                trans_val_data.append(data_val)

                            if self.ref_prior:
                                c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                                prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                            else:
                                c_bin = 0.
                                prior_c = 0.

                            recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, prior_c=prior_c,
                                                                                    eval=True, mask=pruning_mask)
                            loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data,
                                                                                        mu, log_var, qc, c, c_bin)
                            val_loss = loss.data.item()
                            for arm in range(self.arms):
                                val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                            

                    validation_rec_loss[epoch] = val_loss_rec / (batch_indx + 1) / self.arms
                    total_val_loss[epoch] = val_loss / (batch_indx + 1)
                    # print('====> Validation Total Loss: {:.4}, Rec. Loss: {:.4f}'.format(total_val_loss[epoch], validation_rec_loss[epoch]))

#                     print(
#                         f"""        -- Epoch: {epoch} --
# Total Loss:           {train_loss[epoch]:.4f}
# Rec_arm_1:            {train_recon[0, epoch]:.4f}
# Rec_arm_2:            {train_recon[1, epoch]:.4f}
# Joint Loss:           {train_loss_joint[epoch]:.4f}
# Entropy:              {train_entropy[epoch]:.4f}
# Distance:             {train_distance[epoch]:.4f}
# Elapsed Time:         {time.time() - t0:.2f}
# Validation Loss:      {total_val_loss[epoch]:.4f}
# Validation Rec. Loss: {validation_rec_loss[epoch]:.4f}
# """
#                     )
                    print(f" -- EPOCH {str(epoch)} -- ".center((EPOCH_WIDTH // 2) + 5, ' '))
                    print(_label('Total Loss:'), _value(train_loss[epoch]))
                    print(_label('Rec_arm_1:'), _value(train_recon[0, epoch]))
                    print(_label('Rec_arm_2:'), _value(train_recon[1, epoch]))
                    print(_label('Joint Loss:'), _value(train_loss_joint[epoch]))
                    print(_label('Entropy:'), _value(train_entropy[epoch]))
                    print(_label('Distance:'), _value(train_distance[epoch]))
                    print(_label('Elapsed Time:'), _value(time.time() - t0))
                    print(_label('Validation Loss:'), _value(total_val_loss[epoch]))
                    print(_label('Validation Rec. Loss:'), _value(validation_rec_loss[epoch]))

                    pbar.set_postfix({'Total Loss': train_loss[epoch], 'Rec_arm_1': train_recon[0, epoch], 'Rec_arm_2': train_recon[1, epoch], 'Joint Loss': train_loss_joint[epoch], 'Entropy': train_entropy[epoch], 'Distance': train_distance[epoch], 'Elapsed Time': time.time() - t0, 'Validation Loss': total_val_loss[epoch], 'Validation Rec. Loss': validation_rec_loss[epoch]})

                for arm in range(self.arms):
                    prune.remove(self.model.fcc[arm], 'weight')
                    prune.remove(self.model.fcc[arm], 'bias')
                    prune.remove(self.model.fc_mu[arm], 'weight')
                    prune.remove(self.model.fc_sigma[arm], 'weight')
                    prune.remove(self.model.fc6[arm], 'weight')

                trained_model = self.folder + '/model/cpl_mixVAE_model_after_pruning_' + str(pr+1) + '_' + self.current_time + '.pth'
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)
                # plot the learning curve of the network
                fig, ax = plt.subplots()
                ax.plot(range(n_epoch_p), train_loss, label='Training')
                ax.plot(range(n_epoch_p), total_val_loss, label='Validation')
                ax.set_xlabel('# epoch', fontsize=16)
                ax.set_ylabel('loss value', fontsize=16)
                ax.set_title('Learning curve of the cpl-mixVAE for K=' + str(self.categories) + ' and S=' + str(
                    self.state_dim))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.legend()
                ax.figure.savefig(self.folder + '/model/learning_curve_after_pruning_' + str(pr+1) + '_K_' + str(
                    self.categories) + '_' + self.current_time + '.png')
                plt.close("all")
                pr += 1
        
        print('Training is done!')
        return trained_model

    def eval_model(self, data_loader, c_p=0, c_onehot=0):
        """
        run the training of the cpl-mixVAE with the pre-defined parameters/settings
        pcikle used for saving the file

        input args
            data_loader: input data loader
            c_p: the prior categorical variable, only if ref_prior is True.
            c_onehot: the one-hot representation of the prior categorical variable, only if ref_prior is True.

        return
            d_dict: the output dictionary.
        """

        self.model.eval()
        bias = self.model.fcc[0].bias.detach().cpu().numpy()
        pruning_mask = np.where(bias != 0.)[0]
        prune_indx = np.where(bias == 0.)[0]
        max_len = len(data_loader.dataset)
        recon_cell = np.zeros((self.arms, max_len, self.input_dim))
        p_cell = np.zeros((self.arms, max_len, self.input_dim))
        state_sample = np.zeros((self.arms, max_len, self.state_dim))
        state_mu = np.zeros((self.arms, max_len, self.state_dim))
        state_var = np.zeros((self.arms, max_len, self.state_dim))
        z_prob = np.zeros((self.arms, max_len, self.categories))
        z_sample = np.zeros((self.arms, max_len, self.categories))
        data_low = np.zeros((self.arms, max_len, self.lowD_dim))
        state_cat = np.zeros([self.arms, max_len])
        prob_cat = np.zeros([self.arms, max_len])
        if self.ref_prior:
            predicted_label = np.zeros((self.arms+1, max_len))
        else:
            predicted_label = np.zeros((self.arms, max_len))
        data_indx = np.zeros(max_len)
        total_loss_val = []
        total_dist_z = []
        total_dist_qz = []
        total_loss_rec = [[] for a in range(self.arms)]
        total_loglikelihood = [[] for a in range(self.arms)]

        self.model.eval()
        batch_size = data_loader.batch_size

        with torch.no_grad():
            if batch_size > 1:
                for i, (data, data_idx) in enumerate(data_loader):
                    data = data.to(self.device)
                    data_idx = data_idx.to(int)
                    
                    if self.ref_prior:
                        c_bin = torch.FloatTensor(c_onehot[data_idx, :]).to(self.device)
                        prior_c = torch.FloatTensor(c_p[data_idx, :]).to(self.device)
                    else:
                        c_bin = 0.
                        prior_c = 0.

                    trans_data = []
                    for arm in range(self.arms):
                        trans_data.append(data)

                    recon, p_x, r_x, x_low, z_category, state, z_smp, mu, log_sigma, _ = self.model(trans_data, self.temp, prior_c=prior_c, eval=True, mask=pruning_mask)
                    loss, loss_arms, loss_joint, _, dist_z, d_qz, _, _, loglikelihood = self.model.loss(recon, p_x, r_x, trans_data, mu, log_sigma, z_category, z_smp, c_bin)
                    total_loss_val.append(loss.data.item())
                    total_dist_z.append(dist_z.data.item())
                    total_dist_qz.append(d_qz.data.item())

                    if self.ref_prior:
                        predicted_label[0, i * batch_size:min((i + 1) * batch_size, max_len)] = np.argmax(c_p[data_idx, :], axis=1) + 1

                    for arm in range(self.arms):
                        total_loss_rec[arm].append(loss_arms[arm].data.item())
                        total_loglikelihood[arm].append(loglikelihood[arm].data.item())

                    for arm in range(self.arms):
                        state_sample[arm, i * batch_size:min((i + 1) * batch_size, max_len), :] = state[arm].cpu().detach().numpy()
                        state_mu[arm, i * batch_size:min((i + 1) * batch_size, max_len), :] = mu[arm].cpu().detach().numpy()
                        state_var[arm, i * batch_size:min((i + 1) * batch_size, max_len), :] = log_sigma[arm].cpu().detach().numpy()
                        z_encoder = z_category[arm].cpu().data.view(z_category[arm].size()[0], self.categories).detach().numpy()
                        z_prob[arm, i * batch_size:min((i + 1) * batch_size, max_len), :] = z_encoder
                        z_samp = z_smp[arm].cpu().data.view(z_smp[arm].size()[0], self.categories).detach().numpy()
                        z_sample[arm, i * batch_size:min((i + 1) * batch_size, max_len), :] = z_samp
                        data_low[arm,  i * batch_size:min((i + 1) * batch_size, max_len), :] = x_low[arm].detach().cpu().numpy()
                        label = data_idx.numpy().astype(int)
                        data_indx[i * batch_size:min((i + 1) * batch_size, max_len)] = label
                        recon_cell[arm, i * batch_size:min((i + 1) * batch_size, max_len), :] = recon[arm].cpu().detach().numpy()

                        for n in range(z_encoder.shape[0]):
                            state_cat[arm, i * batch_size + n] = np.argmax(z_encoder[n, :]) + 1
                            prob_cat[arm, i * batch_size + n] = np.max(z_encoder[n, :])

                        if self.ref_prior:
                            predicted_label[arm+1, i * batch_size:min((i + 1) * batch_size, max_len)] = np.argmax(z_encoder, axis=1) + 1
                        else:
                            predicted_label[arm, i * batch_size:min((i + 1) * batch_size, max_len)] = np.argmax(z_encoder, axis=1) + 1

            else:
                i = 0
                data, data_idx = data_loader.dataset.tensors
                data = data.to(self.device)
                data_idx = data_idx.to(int)
                if self.ref_prior:
                    c_bin = torch.FloatTensor(c_onehot[data_idx, :]).to(self.device)
                    prior_c = torch.FloatTensor(c_p[data_idx, :]).to(self.device)
                else:
                    c_bin = 0.
                    prior_c = 0.
                trans_data = []
                for arm in range(self.arms):
                    trans_data.append(data)

                recon, p_x, r_x, x_low, z_category, state, z_smp, mu, log_sigma, _ = self.model(trans_data, self.temp, prior_c=prior_c, eval=True, mask=pruning_mask)
                loss, loss_arms, loss_joint, _, dist_z, d_qz, _, _, loglikelihood = self.model.loss(recon, p_x, r_x, trans_data, mu, log_sigma, z_category, z_smp, c_bin)
                total_loss_val = loss.data.item()
                total_dist_z = dist_z.data.item()
                total_dist_qz = d_qz.data.item()
                if self.ref_prior:
                    predicted_label[0, :] = np.argmax(c_p[data_idx, :], axis=1) + 1

                for arm in range(self.arms):
                    total_loss_rec[arm] = loss_arms[arm].data.item()
                    total_loglikelihood[arm] = loglikelihood[arm].data.item()

                for arm in range(self.arms):
                    state_sample[arm, :, :] = state[arm].cpu().detach().numpy()
                    state_mu[arm, :, :] = mu[arm].cpu().detach().numpy()
                    state_var[arm, :, :] = log_sigma[arm].cpu().detach().numpy()
                    z_encoder = z_category[arm].cpu().data.view(z_category[arm].size()[0], self.categories).detach().numpy()
                    z_prob[arm, :, :] = z_encoder
                    z_samp = z_smp[arm].cpu().data.view(z_smp[arm].size()[0], self.categories).detach().numpy()
                    z_sample[arm, :, :] = z_samp
                    data_low[arm, :, :] = x_low[arm].detach().cpu().numpy()
                    label = data_idx.numpy().astype(int)
                    data_indx = label
                    recon_cell[arm, :, :] = recon[arm].cpu().detach().numpy()

                    for n in range(z_encoder.shape[0]):
                        state_cat[arm, n] = np.argmax(z_encoder[n, :]) + 1
                        prob_cat[arm, n] = np.max(z_encoder[n, :])

                    if self.ref_prior:
                        predicted_label[arm+1, :] = np.argmax(z_encoder, axis=1) + 1
                    else:
                        predicted_label[arm, ] = np.argmax(z_encoder, axis=1) + 1


        mean_test_rec = np.zeros(self.arms)
        mean_total_loss_rec = np.zeros(self.arms)
        mean_total_loglikelihood = np.zeros(self.arms)

        for arm in range(self.arms):
            mean_total_loss_rec[arm] = np.mean(np.array(total_loss_rec[arm]))
            mean_total_loglikelihood[arm] = np.mean(np.array(total_loglikelihood[arm]))
 

        d_dict = dict()
        d_dict['state_sample'] = state_sample
        d_dict['state_mu'] = state_mu
        d_dict['state_var'] = state_var
        d_dict['state_cat'] = state_cat
        d_dict['prob_cat'] = prob_cat
        d_dict['total_loss_rec'] = mean_total_loss_rec
        d_dict['total_likelihood'] = mean_total_loglikelihood
        d_dict['total_dist_z'] = np.mean(np.array(total_dist_z))
        d_dict['total_dist_qz'] = np.mean(np.array(total_dist_qz))
        d_dict['mean_test_rec'] = mean_test_rec
        d_dict['predicted_label'] = predicted_label
        d_dict['data_indx'] = data_indx
        d_dict['z_prob'] = z_prob
        d_dict['z_sample'] = z_sample
        d_dict['x_low'] = data_low
        d_dict['recon_c'] = recon_cell
        d_dict['prune_indx'] = prune_indx

        return d_dict


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
        pickle.dump(data, f, protocol=4)
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


# ---------------------------- fsdp ---------------------------- #

    # TODO: make static/function
    def setup(rank, world_size): 
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # TODO: make static/function
    def cleanup():
      dist.destroy_process_group()

    def fsdp(self, train_loader, test_loader, n_epoch, n_epoch_p, c_p=0, c_onehot=0, min_con=0.5, max_prun_it=0, rank=None, world_size=None, sampler=None, logging=False):
      ...

    def check_same_as_train(self):
        ...

    def _prune(self):
        ...
    
    # TODO:
    # [] ref_prior stuff does not work
    # [] fix: batch_size = 1 case
    def val(self, val_loader, device):
        self.model.eval()
        with torch.no_grad():
            val_loss_rec = 0.0
            val_loss = 0.0
            if val_loader.batch_size > 1:
                for i, (data, d), in enumerate(val_loader): # batch index, data, data index
                    data = data.to(self.device)
                    d = d.to(int)
                    trans_val_data = []
                    for arm in range(self.arms):
                        trans_val_data.append(data)

                    if self.ref_prior:
                        c_bin = torch.FloatTensor(c_onehot[d, :]).to(self.device)
                        prior_c = torch.FloatTensor(c_p[d, :]).to(self.device)
                    else:
                        c_bin = 0.
                        prior_c = 0.

                    recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, prior_c=prior_c, eval=True)
                    loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data, mu, log_var, qc, c, c_bin)
                    val_loss += loss.data.item()
                    for arm in range(self.arms):
                        val_loss_rec += loss_rec[arm].data.item() / self.input_dim
            else:
                i = 0
                data, d = val_loader.dataset.tensors
                data = data.to(self.device)
                d = d.to(int)
                trans_val_data = []
                for _ in range(self.arms):
                    trans_val_data.append(data)

                if self.ref_prior:
                    c_bin = torch.FloatTensor(c_onehot[d, :]).to(self.device)
                    prior_c = torch.FloatTensor(c_p[d, :]).to(self.device)
                else:
                    c_bin = 0.
                    prior_c = 0.

                recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data,
                                                                                    temp=self.temp,
                                                                                    prior_c=prior_c, eval=True)
                # loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x,
                #                                                                 trans_val_data, mu, log_var, qc,
                #                                                                 c, c_bin)
                loss, loss_rec, loss_joint, _, _, _, _, _, _  = loss_fn(x=trans_val_data, x_rec=recon_batch, x_succ=p_x, x_disp=r_x, s_mean=mu, s_logvar=log_var, c_pdf=qc, c_samp=c, c_prior=c_bin, A=self.arms, mode=self.mode, is_var=self.is_var, beta=self.beta, eps=self.eps, C=self.categories, lam=self.lam, lam_pc=self.lam_pc, pri=self.ref_prior, device=device)
                val_loss = loss.data.item()
                for a in range(self.arms):
                    val_loss_rec += loss_rec[a].data.item() / self.input_dim

        return {
            'total': val_loss / (i + 1),
            'rec': val_loss_rec / (i + 1) / self.arms
        }
    
    # TODO:
    # [] add "check with train" function
    # [] remove test loader
    # [] remove validation code
    # [] .to(device) vs .to(rank)
    # [] check the loss divisions
    # [] ref_prior stuff does not work
    # [] remove loss function from model
    # [] change val functiong
    # [] add logging for multiple arms
    # [] pass in higher order function for validation
    # [] add saving (look at train())
    # [] check asserts
    # [] make ref_prior version work, too
    # [] pruning
    def _fsdp(self, model, train_loader, val_loader, epochs, n_epoch_p, opt, device, c_p=0, c_onehot=0, min_con=0.5, max_prun_it=0, rank=None, world_size=None, sampler=None):
        assert torch.cuda.is_available()
        model = torch.compile(model).to(device)
        
        # initialized saving arrays
        losses = {
            'total': np.zeros(epochs),
            'rec': np.zeros((self.arms, epochs)),
            'joint': np.zeros(epochs),
            'var': np.zeros(epochs),
            'c_entp': np.zeros(epochs),
            'c_ddist': np.zeros(epochs),
            'c_dist': np.zeros(epochs),
            's_kl': np.zeros((self.arms, self.categories, epochs)),
            'val_total': np.zeros(epochs),
            'val_rec': np.zeros(epochs)
        }

        B = bsize(train_loader)

        pbar = tqdm(range(epochs))
        for epoch in pbar:
            model.train()
            _losses = {
                'total': 0.0,
                'rec': np.zeros(self.arms),
                'joint': 0.0,
                'var': 0.0,
                'c_entp': 0.0,
                'c_ddist': 0.0,
                'c_dist': 0.0,
                's_kl': 0.0
            }
            _val = {
                'total': 0.0,
                'rec': 0.0
            }
            for i, (data, d), in enumerate(train_loader): # batch index, (data, data index)
                data, d = to(data, device), to(d, int)
                    
                # TODO: wtf is this
                trans_data = []
                for _ in range(self.arms):
                    if self.aug_file:
                        noise = torch.randn(B, self.aug_param['num_n'], device=device)
                        _, gen_data = self.netA(data, noise, True, device)
                        if self.aug_param['n_zim'] > 1:
                            data_bin = 0.0 * data
                            data_bin[data > self.eps] = 1.0
                            fake_data = gen_data[:, :self.aug_param['n_features']] * data_bin
                            trans_data.append(fake_data)
                        else:
                            trans_data.append(gen_data)
                    else:
                        trans_data.append(data)

                if self.ref_prior:
                    c_bin = to(torch.FloatTensor(c_onehot[d, :]), device)
                    prior_c = to(torch.FloatTensor(c_p[d, :]), device)
                else:
                    c_bin = 0.0
                    prior_c = 0.0

                opt.zero_grad()
                recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, log_qc = model(x=trans_data, temp=self.temp, prior_c=prior_c)
                _out = loss_fn(x=trans_data, x_rec=recon_batch, x_succ=p_x, x_disp=r_x, s_mean=mu, s_logvar=log_var, c_pdf=qc, c_samp=c, c_prior=c_bin, A=self.arms, mode=self.mode, is_var=self.is_var, beta=self.beta, eps=self.eps, C=self.categories, lam=self.lam, lam_pc=self.lam_pc, pri=self.ref_prior, device=device)
                loss, loss_rec, loss_joint, entropy, dist_c, d_qc, KLD_cont, min_var_0, loglikelihood = _out
                loss.backward()
                opt.step()

                _losses['total'] += item(loss)
                for a in range(self.arms):
                   _losses['rec'][a] += item(loss_rec[a]) / self.input_dim 
                _losses['joint'] += item(loss_joint)
                _losses['var'] += item(min_var_0)
                _losses['c_entp'] += item(entropy)
                _losses['c_ddist'] += item(d_qc)
                _losses['c_dist'] += item(dist_c)
                
            # validation
            _val = self.val(val_loader, device)

            losses['total'][epoch] = _losses['total'] / (i + 1)
            for a in range(self.arms):
                losses['rec'][a][epoch] = _losses['rec'][a] / (i + 1)
            losses['joint'][epoch] = _losses['joint'] / (i + 1)
            losses['var'][epoch] = _losses['var'] / (i + 1)
            losses['c_entp'][epoch] = _losses['c_entp'] / (i + 1)
            losses['c_ddist'][epoch] = _losses['c_ddist'] / (i + 1)
            losses['c_dist'][epoch] = _losses['c_dist'] / (i + 1)
            losses['val_total'][epoch] = _val['total']
            losses['val_rec'][epoch] = _val['rec']
            
            print(f" -- EPOCH {str(epoch)} -- ".center((EPOCH_WIDTH // 2) + 5, ' '))
            print(_label('Total Loss:'), _value(losses['total'][epoch]))
            print(_label('Rec_arm_1:'), _value(losses['rec'][0][epoch]))
            print(_label('Rec_arm_2:'), _value(losses['rec'][1][epoch]))
            print(_label('Joint Loss:'), _value(losses['joint'][epoch]))
            print(_label('Entropy:'), _value(losses['c_entp'][epoch]))
            print(_label('Distance:'), _value(losses['c_dist'][epoch]))
            print(_label('Validation Loss:'), _value(losses['val_total'][epoch]))
            print(_label('Validation Rec. Loss:'), _value(losses['val_rec'][epoch]))
            pbar.set_postfix({'Total Loss': losses['total'][epoch],
                              'Rec_arm_1': losses['rec'][0][epoch],
                              'Rec_arm_2': losses['rec'][1][epoch],
                              'Joint Loss': losses['joint'][epoch],
                              'Entropy': losses['c_entp'][epoch],
                              'Distance': losses['c_dist'][epoch],
                              'Validation Loss': losses['val_total'][epoch],
                              'Validation Rec. Loss': losses['val_rec'][epoch]})
        # ------------------- Pruning -------------------
#         masks = {
#             'bias': torch.ones(self.categories).to(device),
#             'weight': torch.ones((self.categories, self.lowD_dim)).to(device),
#             'fc_mu': torch.ones((self.state_dim, self.categories + self.lowD_dim)).to(device),
#             'fc_sigma': torch.ones((self.state_dim, self.categories + self.lowD_dim)).to(device),
#             'f6': torch.ones((self.lowD_dim, self.state_dim + self.categories)).to(device)
#         }

#         bias_mask = torch.ones(self.categories).to(device)
#         weight_mask = torch.ones((self.categories, self.lowD_dim)).to(device)
#         fc_mu = torch.ones((self.state_dim, self.categories + self.lowD_dim)).to(device)
#         fc_sigma = torch.ones((self.state_dim, self.categories + self.lowD_dim)).to(device)
#         f6_mask = torch.ones((self.lowD_dim, self.state_dim + self.categories)).to(device)

#         bias = model.fcc[0].bias.detach().cpu().numpy()
#         prune_indx = []
#         if n_epoch_p > 0:
#             # initialized pruning parameters of the layer of the discrete variable
#             bias = model.fcc[0].bias.detach().cpu().numpy()
#             pruning_mask = np.where(bias != 0.)[0]
#             prune_indx = np.where(bias == 0.)[0]
#             stop_prune = False
#         else:
#             stop_prune = True

#         pr = self.n_pr
#         ind = []
#         while not stop_prune:
#             predicted_label = np.zeros((self.arms, len(train_loader.dataset)))

#             # Assessment over all dataset
#             model.eval()
#             with torch.no_grad():
#                 for i, (data, d_idx) in enumerate(train_loader):
#                     data = data.to(device)
#                     d_idx = d_idx.to(int)
#                     trans_data = []
#                     for arm in range(self.arms):
#                         trans_data.append(data)

#                     if self.ref_prior:
#                         c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(device)
#                         prior_c = torch.FloatTensor(c_p[d_idx, :]).to(device)
#                     else:
#                         c_bin = 0.
#                         prior_c = 0.

#                     recon, p_x, r_x, x_low, z_category, state, z_smp, mu, log_sigma, _ = model(trans_data, self.temp, prior_c, mask=pruning_mask, eval=True)

#                     for arm in range(self.arms):
#                         z_encoder = z_category[arm].cpu().data.view(z_category[arm].size()[0], self.categories).detach().numpy()
#                         predicted_label[arm, i * B:min((i + 1) * B, len(train_loader.dataset))] = np.argmax(z_encoder, axis=1)

#             c_agreement = []
#             for arm_a in range(self.arms):
#                 pred_a = predicted_label[arm_a, :]
#                 for arm_b in range(arm_a + 1, self.arms):
#                     pred_b = predicted_label[arm_b, :]
#                     armA_vs_armB = np.zeros((self.categories, self.categories))

#                     for samp in range(pred_a.shape[0]):
#                         armA_vs_armB[pred_a[samp].astype(int), pred_b[samp].astype(int)] += 1

#                     num_samp_arm = []
#                     for ij in range(self.categories):
#                         sum_row = armA_vs_armB[ij, :].sum()
#                         sum_column = armA_vs_armB[:, ij].sum()
#                         num_samp_arm.append(max(sum_row, sum_column))

#                     armA_vs_armB = np.divide(armA_vs_armB, np.array(num_samp_arm), out=np.zeros_like(armA_vs_armB),
#                                              where=np.array(num_samp_arm) != 0)
#                     c_agreement.append(np.diag(armA_vs_armB))

#             c_agreement = np.mean(c_agreement, axis=0)
#             agreement = c_agreement[pruning_mask]
#             if (np.min(agreement) <= min_con) and pr < max_prun_it:
#                 if pr > 0:
#                     ind_min = pruning_mask[np.argmin(agreement)]
#                     ind_min = np.array([ind_min])
#                     ind = np.concatenate((ind, ind_min))
#                 else:
#                     ind_min = pruning_mask[np.argmin(agreement)]
#                     if len(prune_indx) > 0:
#                         ind_min = np.array([ind_min])
#                         ind = np.concatenate((prune_indx, ind_min))
#                     else:
#                         ind.append(ind_min)
#                     ind = np.array(ind)

#                 ind = ind.astype(int)
#                 bias_mask[ind] = 0.
#                 weight_mask[ind, :] = 0.
#                 fc_mu[:, self.lowD_dim + ind] = 0.
#                 fc_sigma[:, self.lowD_dim + ind] = 0.
#                 f6_mask[:, ind] = 0.
#                 stop_prune = False
#             else:
#                 stop_prune = True

#             if not stop_prune:
#                 bias = bias_mask.detach().cpu().numpy()
#                 pruning_mask = np.where(bias != 0.)[0]
#                 train_loss = np.zeros(n_epoch_p)
#                 validation_rec_loss = np.zeros(n_epoch_p)
#                 total_val_loss = np.zeros(n_epoch_p)
#                 train_loss_joint = np.zeros(n_epoch_p)
#                 train_entropy = np.zeros(n_epoch_p)
#                 train_distance = np.zeros(n_epoch_p)
#                 train_minVar = np.zeros(n_epoch_p)
#                 train_log_distance = np.zeros(n_epoch_p)
#                 train_recon = np.zeros((self.arms, n_epoch_p))
#                 train_loss_KL = np.zeros((self.arms, self.categories, n_epoch_p))

#                 for arm in range(self.arms):
#                     prune.custom_from_mask(model.fcc[arm], 'weight', mask=weight_mask)
#                     prune.custom_from_mask(model.fcc[arm], 'bias', mask=bias_mask)
#                     prune.custom_from_mask(model.fc_mu[arm], 'weight', mask=fc_mu)
#                     prune.custom_from_mask(model.fc_sigma[arm], 'weight', mask=fc_sigma)
#                     prune.custom_from_mask(model.fc6[arm], 'weight', mask=f6_mask)

#                 pbar = tqdm(range(n_epoch_p))
#                 for epoch in pbar:
#                     # training
#                     train_loss_val = 0
#                     train_jointloss_val = 0
#                     train_dqz = 0
#                     log_dqz = 0
#                     entr = 0
#                     var_min = 0
#                     t0 = time.time()
#                     train_loss_rec = np.zeros(self.arms)
#                     train_KLD_cont = np.zeros((self.arms, self.categories))
#                     ti = np.zeros(len(train_loader))
#                     model.train()
#                     # training
#                     for batch_indx, (data, d_idx), in enumerate(train_loader):
#                         # for data in train_loader:
#                         data = data.to(self.device)
#                         d_idx = d_idx.to(int)
#                         data_bin = 0. * data
#                         data_bin[data > 0.0] = 1.
#                         trans_data = []
#                         trans_data.append(data)
#                         tt = time.time()
#                         for arm in range(self.arms-1):
#                             if self.aug_file:
#                                 noise = torch.randn(B, self.aug_param['num_n']).to(device)
#                                 _, gen_data = self.netA(data, noise, True, device)
#                                 if self.aug_param['n_zim'] > 1:
#                                     data_bin = 0. * data
#                                     data_bin[data > self.eps] = 1.
#                                     fake_data = gen_data[:, :self.aug_param['n_features']] * data_bin
#                                     trans_data.append(fake_data)
#                                 else:
#                                     trans_data.append(gen_data)
#                             else:
#                                 trans_data.append(data)

#                         if self.ref_prior:
#                             c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(device)
#                             prior_c = torch.FloatTensor(c_p[d_idx, :]).to(device)
#                         else:
#                             c_bin = 0.
#                             prior_c = 0.

#                         opt.zero_grad()
#                         recon_batch, p_x, r_x, x_low, qz, s, z, mu, log_var, log_qz = model(trans_data, self.temp, prior_c, mask=pruning_mask)
#                         loss, loss_rec, loss_joint, entropy, dist_z, d_qz, KLD_cont, min_var_0, _ = model.loss(recon_batch, p_x, r_x,
#                                                                                         trans_data, mu, log_var, qz, z, c_bin)

#                         loss.backward()
#                         opt.step()
#                         ti[batch_indx] = time.time() - tt
#                         train_loss_val += loss.data.item()
#                         train_jointloss_val += loss_joint
#                         train_dqz += d_qz
#                         log_dqz += dist_z
#                         entr += entropy
#                         var_min += min_var_0.data.item()

#                         for arm in range(self.arms):
#                             train_loss_rec[arm] += loss_rec[arm].data.item() / self.input_dim

#                     train_loss[epoch] = train_loss_val / (batch_indx + 1)
#                     train_loss_joint[epoch] = train_jointloss_val / (batch_indx + 1)
#                     train_distance[epoch] = train_dqz / (batch_indx + 1)
#                     train_entropy[epoch] = entr / (batch_indx + 1)
#                     train_log_distance[epoch] = log_dqz / (batch_indx + 1)
#                     train_minVar[epoch] = var_min / (batch_indx + 1)

#                     for arm in range(self.arms):
#                         train_recon[arm, epoch] = train_loss_rec[arm] / (batch_indx + 1)
#                         for c in range(self.categories):
#                             train_loss_KL[arm, c, epoch] = train_KLD_cont[arm, c] / (batch_indx + 1)

#                     # validation
#                     model.eval()
#                     with torch.no_grad():
#                         val_loss_rec = 0.
#                         val_loss = 0.
#                         if val_loader.batch_size > 1:
#                             for batch_indx, (data_val, d_idx), in enumerate(val_loader):
#                                 d_idx = d_idx.to(int)
#                                 data_val = data_val.to(device)
                                    
#                                 trans_val_data = []
#                                 for arm in range(self.arms):
#                                     trans_val_data.append(data_val)

#                                 if self.ref_prior:
#                                     c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(device)
#                                     prior_c = torch.FloatTensor(c_p[d_idx, :]).to(device)
#                                 else:
#                                     c_bin = 0.
#                                     prior_c = 0.

#                                 recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = model(x=trans_val_data, temp=self.temp, prior_c=prior_c,
#                                                                                         eval=True, mask=pruning_mask)
#                                 loss, loss_rec, loss_joint, _, _, _, _, _, _ = model.loss(recon_batch, p_x, r_x, trans_val_data,
#                                                                                             mu, log_var, qc, c, c_bin)
#                                 val_loss += loss.data.item()
#                                 for arm in range(self.arms):
#                                     val_loss_rec += loss_rec[arm].data.item() / self.input_dim
#                         else:
#                             batch_indx = 0
#                             data_val, d_idx = val_loader.dataset.tensors
#                             data_val = data_val.to(device)
#                             d_idx = d_idx.to(int)
#                             trans_val_data = []
#                             for arm in range(self.arms):
#                                 trans_val_data.append(data_val)

#                             if self.ref_prior:
#                                 c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(device)
#                                 prior_c = torch.FloatTensor(c_p[d_idx, :]).to(device)
#                             else:
#                                 c_bin = 0.
#                                 prior_c = 0.

#                             recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = model(x=trans_val_data, temp=self.temp, prior_c=prior_c,
#                                                                                     eval=True, mask=pruning_mask)
#                             loss, loss_rec, loss_joint, _, _, _, _, _, _ = model.loss(recon_batch, p_x, r_x, trans_val_data,
#                                                                                         mu, log_var, qc, c, c_bin)
#                             val_loss = loss.data.item()
#                             for arm in range(self.arms):
#                                 val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                            

#                     validation_rec_loss[epoch] = val_loss_rec / (batch_indx + 1) / self.arms
#                     total_val_loss[epoch] = val_loss / (batch_indx + 1)

#                     print(
#                         f"""        -- Epoch: {epoch} --
# Total Loss:           {train_loss[epoch]:.4f}
# Rec_arm_1:            {train_recon[0, epoch]:.4f}
# Rec_arm_2:            {train_recon[1, epoch]:.4f}
# Joint Loss:           {train_loss_joint[epoch]:.4f}
# Entropy:              {train_entropy[epoch]:.4f}
# Distance:             {train_distance[epoch]:.4f}
# Elapsed Time:         {time.time() - t0:.2f}
# Validation Loss:      {total_val_loss[epoch]:.4f}
# Validation Rec. Loss: {validation_rec_loss[epoch]:.4f}
# """
#                     )
#                     pbar.set_postfix({'Total Loss': train_loss[epoch],
#                                       'Rec_arm_1': train_recon[0, epoch],
#                                       'Rec_arm_2': train_recon[1, epoch],
#                                       'Joint Loss': train_loss_joint[epoch],
#                                       'Entropy': train_entropy[epoch],
#                                       'Distance': train_distance[epoch], 
#                                       'Elapsed Time': time.time() - t0, 
#                                       'Validation Loss': total_val_loss[epoch], 
#                                       'Validation Rec. Loss': 
#                                           validation_rec_loss[epoch]})
#                 for a in range(self.arms):
#                     prune.remove(model.fcc[a], 'weight')
#                     prune.remove(model.fcc[a], 'bias')
#                     prune.remove(model.fc_mu[a], 'weight')
#                     prune.remove(model.fc_sigma[a], 'weight')
#                     prune.remove(model.fc6[a], 'weight')
#                 pr += 1 
        return losses
        
    def test_fsdp(self, test_loader, n_epoch, n_epoch_p, c_p=0, c_onehot=0, min_con=.5, max_prun_it=0, rank=None, world_size=None):
        pass 

    def fsdp_main(rank, world_size, args):
        pass

def pprint_training_dict(fsdp_training_dict):
    ...

def fst(xs): 
    return getitem(xs, 0)

fsts = partial(map, fst)