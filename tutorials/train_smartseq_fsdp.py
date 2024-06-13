import argparse
import os
from mmidas.nn_model import mixVAE_Config
from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.utils.tools import get_paths
from mmidas.utils.dataloader import load_data, get_loaders

from torchinfo import summary

def main(): 
    config = mixVAE_Config()
    print(config)


    # toml_file = 'pyproject.toml'
    # sub_file = 'smartseq_files'
    # config = get_paths(toml_file=toml_file, sub_file=sub_file)
    # data_path = config['paths']['main_dir'] / config['paths']['data_path']
    # data_file = data_path / config[sub_file]['anndata_file']


if __name__ == '__main__': 
    main()