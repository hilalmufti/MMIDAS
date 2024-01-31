import os
import sys, pickle
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from utils.analysis_cells_tree import HTree, dend_json_to_df
from config.path import DSPRITESPATH
from utils.reader_op import read_npy
from utils.datamanager import DspritesManager

import numpy as np

def dsprites_manager():

    data_dir = '/allen/programs/celltypes/workgroups/mousecelltypes/Yeganeh/CTX-HIP/mouse/brain_map_10x/'
    local_data_dir = '/home/yeganeh/Remote-AI/CTX-HIP/mouse/brain_map_10x/'
    data_gaba = "gaba_cascadeVAE.p" #"GABAergic_isoCTX_nGene_10000.h5ad"
    data_glum = "Glutamatergic_isoCTX_nGene_10000.h5ad"
    data_file = local_data_dir + data_gaba
    dm = DspritesManager(data_file)
    return dm

#dsprites_ndarray_co1sh3sc6or40x32y32_64x64