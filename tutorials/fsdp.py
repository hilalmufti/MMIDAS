import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import mmidas
from mmidas.nn_model import mixVAE_model, loss_fn
from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.utils.tools import get_paths
from mmidas.utils.dataloader import load_data, get_loaders

import os

import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


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

import datetime

def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=120))

def cleanup():
  dist.destroy_process_group()

def memory_stats():
    print(torch.cuda.memory_allocated()/1024**2)
    print(torch.cuda.memory_cached()/1024**2)

def train():
  ...

def test():
  ...

# Define the main function
# [] each arm should probably be in its own fsdp unit
# [] find best size-based autowrap policy? 
# [] maybe write an algorithm to find best autowrap policy
# [] do test, too
def main(rank, world_size, args):
    n_categories = args.n_categories
    n_arm = args.n_arm
    state_dim = args.state_dim
    latent_dim = args.latent_dim
    fc_dim = args.fc_dim
    n_epoch = args.n_epoch
    n_epoch_p = args.n_epoch_p
    min_con = args.min_con
    max_prun_it = args.max_prun_it
    batch_size = args.batch_size
    lam = args.lam
    lam_pc = args.lam_pc
    loss_mode = args.loss_mode
    p_drop = args.p_drop
    s_drop = args.s_drop
    lr = args.lr
    temp = args.temp
    n_run = args.n_run
    device = args.device
    hard = args.hard
    tau = args.tau
    variational = args.variational
    ref_pc = args.ref_pc
    augmentation = args.augmentation
    pretrained_model = args.pretrained_model
    n_pr = args.n_pr
    beta = args.beta
      
    setup(rank, world_size)

    print('finished')

    toml_file = 'pyproject.toml'
    sub_file = 'smartseq_files'
    config = get_paths(toml_file=toml_file, sub_file=sub_file)
    data_path = config['paths']['main_dir'] / config['paths']['data_path']
    data_file = data_path / config[sub_file]['anndata_file']

    folder_name = f'run_{n_run}_K_{n_categories}_Sdim_{state_dim}_aug_{augmentation}_lr_{lr}_n_arm_{n_arm}_nbatch_{batch_size}' + \
                f'_train.ipynb_nepoch_{n_epoch}_nepochP_{n_epoch_p}'
    saving_folder = config['paths']['main_dir'] / config['paths']['saving_path']
    saving_folder = saving_folder / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    saving_folder = str(saving_folder)

    if augmentation:
        aug_file = config['paths']['main_dir'] / config[sub_file]['aug_model']
    else:
        aug_file = ''
    
    if pretrained_model:
        trained_model = config['paths']['main_dir'] / config[sub_file]['trained_model']
    else:
        trained_model = ''

    data = load_data(datafile=data_file)
    trainloader, testloader, _, = get_loaders(dataset=data['log1p'], batch_size=batch_size)

    cplMixVAE = cpl_mixVAE(saving_folder=saving_folder, device=rank)
    cplMixVAE.init(categories=n_categories,
                          state_dim=state_dim,
                          input_dim=data['log1p'].shape[1],
                          fc_dim=fc_dim,
                          lowD_dim=latent_dim,
                          x_drop=p_drop,
                          s_drop=s_drop,
                          lr=lr,
                          arms=n_arm,
                          temp=temp,
                          hard=hard,
                          tau=tau,
                          lam=lam,
                          lam_pc=lam_pc,
                          beta=beta,
                          ref_prior=ref_pc,
                          variational=variational,
                          trained_model=trained_model,
                          n_pr=n_pr,
                          mode=loss_mode)

    # -- fsdp -- 

    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True) 
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = cplMixVAE.model.to(rank)
    # model = model.to(rank)
    # model = FSPD(model)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    init_start_event.record()
    losses = cplMixVAE._fsdp(model,
                             train_loader=trainloader,
                             val_loader=testloader,
                            # epochs=n_epoch,
                            epochs=3,
                             n_epoch_p=n_epoch_p,
                             c_p=data['c_p'],
                             c_onehot=data['c_onehot'],
                             min_con=min_con,
                             opt=opt,
                             device = rank,
                            # device=cplMixVAE.device,
                             rank=rank,
                             world_size=world_size)
    init_end_event.record()

    if rank == 0:
      print(f"Training time: {init_start_event.elapsed_time(init_end_event)}")
      print(f"{model}")

    cleanup()


    # model_file = cplMixVAE.train(train_loader=trainloader,
    #                             test_loader=testloader,
    #                             n_epoch=n_epoch,
    #                             n_epoch_p=n_epoch_p,
    #                             c_onehot=data['c_onehot'],
    #                             c_p=data['c_p'],
    #                             min_con=min_con,
    #                             max_prun_it=max_prun_it)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_categories", default=120, type=int, help="(maximum) number of cell types")
  parser.add_argument("--state_dim", default=2, type=int, help="state variable dimension")
  parser.add_argument("--n_arm", default=2, type=int,  help="number of mixVAE arms for each modalities")
  parser.add_argument("--temp",  default=1, type=float, help="gumbel-softmax temperature")
  parser.add_argument("--tau",  default=.005, type=float, help="softmax temperature")
  parser.add_argument("--beta",  default=.01, type=float, help="KL regularization parameter")
  parser.add_argument("--lam",  default=1, type=float, help="coupling factor")
  parser.add_argument("--lam_pc",  default=1, type=float, help="coupling factor for ref arm")
  parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
  parser.add_argument("--n_epoch", default=10000, type=int, help="Number of epochs to train")
  parser.add_argument("--n_epoch_p", default=10000, type=int, help="Number of epochs to train pruning algorithm")
  parser.add_argument("--min_con", default=.99, type=float, help="minimum consensus")
  parser.add_argument("--max_prun_it", default=50, type=int, help="maximum number of pruning iteration")
  parser.add_argument("--ref_pc", default=False, type=bool, help="path of the data augmenter")
  parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
  parser.add_argument("--batch_size", default=5000, type=int, help="batch size")
  parser.add_argument("--variational", default=True, type=bool, help="enable variational mode")
  parser.add_argument("--augmentation", default=False, type=bool, help="enable VAE-GAN augmentation")
  parser.add_argument("--lr", default=.001, type=float, help="learning rate")
  parser.add_argument("--p_drop", default=0.5, type=float, help="input probability of dropout")
  parser.add_argument("--s_drop", default=0.2, type=float, help="state probability of dropout")
  parser.add_argument("--pretrained_model", default=False, type=bool, help="use pretrained model")
  parser.add_argument("--n_pr", default=0, type=int, help="number of pruned categories in case of using a pretrained model")
  parser.add_argument("--loss_mode", default='MSE', type=str, help="loss mode, MSE or ZINB")
  parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
  parser.add_argument("--hard", default=False, type=bool, help="hard encoding")
  parser.add_argument("--device", default='cuda', type=str, help="computing device, either 'cpu' or 'cuda'.")

  args = parser.parse_args()


  WORLD_SIZE = torch.cuda.device_count()
  # main(0, WORLD_SIZE, args)

  mp.spawn(main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
