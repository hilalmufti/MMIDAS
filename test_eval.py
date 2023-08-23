import os, glob, pickle
import scipy.io as sio
import numpy as np
from utils.tree_based_analysis import get_merged_types
from utils.training import train_cplmixVAE
from utils.eval_models import eval_mixmodel
from utils.cluster_analysis import K_selection
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


n_run = 11
n_categories = 115
state_dim = 2
n_arm = 1
tau = .005
fc_dim = 100
latent_dim = 10
lr = 0.001
lams = [100.]
p_drop = .5
batch_size = 1000
n_epoch = 20000
n_epoch_p = 0
n_gene = 5000
n_aug_smp = 0
device = None

cons_lam, recon_loss = [], []
path = os.getcwd()
data_path = path + '/data/Mouse-V1-ALM-20180520_cpmtop10k_cpm_withCL.mat'

for lam in lams:
    folder_name = f'run_{n_run}_bin_K_{n_categories}_Sdim_{state_dim}_p_drop_{p_drop}_fc_dim_100_n_aug_{n_aug_smp}_' + \
                  f'lr_{lr}_n_arm_{n_arm}_tau_{tau}_lam_{lam}_nbatch_{batch_size}_nepoch_{n_epoch}_nepochP_{n_epoch_p}'
    saving_folder = path + "/results/cplmixVAE/"
    saving_folder = saving_folder + str(folder_name)

    model_order = n_categories
    models = glob.glob(saving_folder + '/model/cpl_mixVAE_model_*')
    cpl_mixVAE, data, outcome = eval_mixmodel(n_categories=n_categories,
                                              state_dim=state_dim,
                                              n_arm=n_arm,
                                              latent_dim=latent_dim,
                                              fc_dim=fc_dim,
                                              tau=tau,
                                              n_gene=n_gene,
                                              saving_folder=saving_folder,
                                              data_file=data_path,
                                              device=device,
                                              models=models)

    plt.figure(figsize=[10, 10])
    ax = plt.gca()
    im = ax.imshow(outcome['consensus'][0], cmap='binary', vmax=1)
    plt.xlabel('Ref. arm', fontsize=20)
    plt.ylabel('arm 1', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.title('consensus, |c|= ' + str(outcome['consensus'][0].shape[0]), fontsize=16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im, cax=cax)
    cons_lam.append(np.sum(np.diag(outcome['consensus'][0])) / model_order)
    recon_loss.append(outcome['recon_loss'][0])

print(cons_lam)