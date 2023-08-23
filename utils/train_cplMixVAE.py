import argparse
import os
from utils.training import train_cplmixVAE
from utils.config import load_config
from utils.dataloader import load_data
import pickle
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np

# AD_MTG_norm_L4-IT_ngene_7787.p
parser = argparse.ArgumentParser()
parser.add_argument("--n_categories", default=100, type=int, help="number of cell types, e.g. 120")
parser.add_argument("--state_dim", default=3, type=int, help="state variable dimension")
parser.add_argument("--n_arm", default=2, type=int,  help="number of mixVAE arms for each modalities")
parser.add_argument("--temp",  default=1, type=float, help="gumbel-softmax temperature")
parser.add_argument("--tau",  default=0.005, type=float, help="softmax temperature")
parser.add_argument("--beta",  default=1, type=float, help="KL regularization parameter")
parser.add_argument("--lam",  default=1, type=float, help="coupling factor")
parser.add_argument("--lam_pc",  default=1000, type=float, help="coupling factor for ref arm")
parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
parser.add_argument("--n_epoch", default=10000, type=int, help="Number of epochs to train")
parser.add_argument("--n_epoch_p", default=1000, type=int, help="Number of epochs to train pruning algorithm")
parser.add_argument("--min_con", default=.99, type=float, help="minimum consensus")
parser.add_argument("--min_density", default=10, type=int, help="minimum number of samples in a class")
parser.add_argument("--ref_pc", default=False, type=bool, help="path of the data augmenter")
parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
parser.add_argument("--batch_size", default=5000, type=int, help="batch size")
parser.add_argument("--subclass", default='gaba', type=str, help="cell subclass including gaba and glum")
parser.add_argument("--variational", default=False, type=bool, help="enable variational mode")
parser.add_argument("--augmentation", default=True, type=bool, help="enable VAE-GAN augmentation")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--n_gene", default=0, type=int, help="number of genes")
parser.add_argument("--p_drop", default=0.5, type=float, help="input probability of dropout")
parser.add_argument("--s_drop", default=0.2, type=float, help="state probability of dropout")
parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
parser.add_argument("--hard", default=False, type=bool, help="hard encoding")
parser.add_argument("--device", default=0, type=int, help="gpu device, use None for cpu")


def main(n_categories, n_arm, state_dim, latent_dim, fc_dim, n_epoch, n_epoch_p, min_con, min_density, batch_size, subclass,
         p_drop, s_drop, lr, temp, n_run, device, hard, tau, variational, ref_pc, augmentation, n_gene, lam, lam_pc, beta):

    paths = load_config(config_file='config.toml')
    saving_folder = paths['package_dir'] / paths['saving_folder_cplmix']
    data_file = paths['package_dir'] / paths['data_' + subclass]
    folder_name = f'{subclass}_run_{n_run}_K_{n_categories}_Sdim_{state_dim}_aug_{augmentation}_p_drop_{p_drop}_fc_dim_{fc_dim}_temp_{temp}_' + \
                  f'lr_{lr}_n_arm_{n_arm}_tau_{tau}_lam_{lam}_nbatch_{batch_size}_nepoch_{n_epoch}_nepochP_{n_epoch_p}'

    if augmentation:
        aug_file = paths['package_dir'] / paths['saving_folder_augmenter'] / paths['aug_file']
    else:
        aug_file = ''

    saving_folder = saving_folder / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    saving_folder = str(saving_folder)

    data = load_data(data_file)

    if n_gene == 0:
        n_gene = data['log1p'].shape[1]

    cpl_mixVAE = train_cplmixVAE(saving_folder=saving_folder,
                                 device=device,
                                 aug_file=aug_file,
                                 n_feature=n_gene)
    alldata_loader, train_loader, validation_loader, test_loader = cpl_mixVAE.getdata(dataset=data['log1p'],
                                                                                      label=data['cluster_order'],
                                                                                      batch_size=batch_size)
    cpl_mixVAE.init_model(n_categories=n_categories,
                          state_dim=state_dim,
                          input_dim=data['log1p'].shape[1],
                          fc_dim=fc_dim,
                          lowD_dim=latent_dim,
                          x_drop=p_drop,
                          s_drop=s_drop,
                          lr=lr,
                          n_arm=n_arm,
                          temp=temp,
                          hard=hard,
                          tau=tau,
                          lam=lam,
                          lam_pc=lam_pc,
                          beta=beta,
                          variational=variational,
                          ref_prior=ref_pc)

    cpl_mixVAE.run(train_loader, test_loader, validation_loader, alldata_loader, n_epoch, n_epoch_p, data['c_onehot'], min_con, min_density)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
