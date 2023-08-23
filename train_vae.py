import argparse
import os
from utils.training_VAE import train_VAE
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
parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
parser.add_argument("--n_epoch", default=10, type=int, help="Number of epochs to train")
parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
parser.add_argument("--batch_size", default=1000, type=int, help="batch size")
parser.add_argument("--variational", default=True, type=bool, help="enable variational mode")
parser.add_argument("--lr", default=.001, type=float, help="learning rate")
parser.add_argument("--n_gene", default=5000, type=int, help="number of genes")
parser.add_argument("--p_drop", default=0.5, type=float, help="input probability of dropout")
parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
parser.add_argument("--device", default=None, type=int, help="gpu device, use None for cpu")


def main(latent_dim, fc_dim, n_epoch, batch_size, p_drop, lr, n_run, device, variational, n_gene):

    paths = load_config(config_file='config.toml')
    saving_folder = paths['package_dir'] / paths['saving_folder_vae']
    data_file = paths['package_dir'] / paths['data']
    folder_name = f'run_{n_run}_z_dim_{latent_dim}_fc_dim_{fc_dim}_lr_{lr}_nbatch_{batch_size}_nepoch_{n_epoch}'

    saving_folder = saving_folder / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    saving_folder = str(saving_folder)

    data = load_data(file=data_file, n_gene=n_gene, ref_genes=True)

    ae = train_VAE(saving_folder=saving_folder, device=device)
    alldata_loader, train_loader, validation_loader, test_loader = ae.getdata(dataset=data['log1p'],
                                                                                      label=data['cluster'],
                                                                                      batch_size=batch_size)

    ae.init_model(input_dim=data['log1p'].shape[1],
                  fc_dim=fc_dim,
                  lowD_dim=latent_dim,
                  x_drop=p_drop,
                  lr=lr,
                  variational=variational)

    ae.run(train_loader=train_loader,
                   test_loader=test_loader,
                   validation_loader=validation_loader,
                   alldata_loader=alldata_loader,
                   n_epoch=n_epoch)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
