import os
import sys
import anndata
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from utils.general_class import DatamanagerPlugin
import numpy as np
import pickle


class DspritesManager(DatamanagerPlugin):
    def __init__(self, dataset):

        # key_list = ['log1p', 'gene_id', 'sample_name', 'cluster_color', 'cluster_order', 'cluster_label', 'class_color',
        #             'class_order', 'class_label', 'subclass_color', 'subclass_order', 'subclass_label', 'sex_id',
        #             'donor_sex_label', 'region_color', 'region_id', 'region_label', 'cortical_layer_label']
        # adata = anndata.read_h5ad(dataset)
        # print('data is loaded.')
        #
        # data = dict()
        # data['log1p'] = adata.X.todense()
        # features = adata.var.keys()
        # data['gene_id'] = adata.var[features].values
        # anno_key = adata.obs.keys()
        # for key in anno_key:
        #     data[key] = adata.obs[key].values
        #
        # all_key = list(data.keys())
        # for key in all_key:
        #     if key not in key_list:
        #         del data[key]
        #
        # # remove cells "CR", "NP PPP", "Ndnf", and "Pax"
        # ind = np.where(data['cluster_label'] == '1_CR')[0]
        # for key in data:
        #     if key not in ['gene_id']:
        #         data[key] = np.delete(data[key], ind, axis=0)
        #
        # ind = np.where(data['cluster_label'] == '354_NP PPP')[0]
        # for key in data:
        #     if key not in ['gene_id']:
        #         data[key] = np.delete(data[key], ind, axis=0)
        #
        # ind = np.where(data['cluster_label'] == '22_Ndnf HPF')[0]
        # for key in data:
        #     if key not in ['gene_id']:
        #         data[key] = np.delete(data[key], ind, axis=0)
        #
        # ind = np.where(data['cluster_label'] == '23_Ndnf HPF')[0]
        # for key in data:
        #     if key not in ['gene_id']:
        #         data[key] = np.delete(data[key], ind, axis=0)
        #
        # ind = np.where(data['cluster_label'] == '17_Pax6')[0]
        # for key in data:
        #     if key not in ['gene_id']:
        #         data[key] = np.delete(data[key], ind, axis=0)
        #
        # ind = np.where(data['cluster_label'] == '18_Pax6')[0]
        # for key in data:
        #     if key not in ['gene_id']:
        #         data[key] = np.delete(data[key], ind, axis=0)
        #
        # ind = np.where(data['cluster_label'] == '19_Pax6')[0]
        # for key in data:
        #     if key not in ['gene_id']:
        #         data[key] = np.delete(data[key], ind, axis=0)
        #
        # uniq_clusters = np.unique(data['cluster_label'])
        #
        # count = np.zeros(len(uniq_clusters))
        # for it, tt in enumerate(uniq_clusters):
        #     count[it] = sum(data['cluster_label'] == tt)
        #
        # uniq_clusters = uniq_clusters[count >= 10]
        # subclass_ind = []
        # for tt in uniq_clusters:
        #     subclass_ind.append(np.array([i for i in range(len(data['cluster_label'])) if tt == data['cluster_label'][i]]))
        #
        # subclass_ind = np.concatenate(subclass_ind)
        # ref_len = len(data['cluster_label'])
        #
        # for k in key_list:
        #     if k == 'log1p':
        #         data[k] = np.array(data[k])[subclass_ind, :]
        #     elif k not in ['gene_id', 'log1p']:
        #         data[k] = np.array(data[k])[subclass_ind]
        #
        # data['cluster_id'] = np.zeros(len(data['cluster_order']))
        # for ic, cls_ord in enumerate(np.unique(data['cluster_label'])):
        #     data['cluster_id'][data['cluster_label'] == cls_ord] = int(ic + 1)
        with open(dataset, 'rb') as f:
            data = pickle.load(f)

        print(len(np.unique(data['cluster_label'])), data['log1p'].shape, len(data['gene_id']))

        self.n_genes = len(data['gene_id'])
        self.image = data['log1p']
        self.bin_image = np.copy(self.image)
        self.bin_image[self.bin_image > 0] = 1
        super().__init__(ndata=self.image.shape[0])
        print(f'Dataset shape: {self.image.shape}')

    def print_shape(self):
        print("Image shape : {}({}, max = {}, min = {})".format(self.image.shape, self.image.dtype, np.amax(self.image), np.amin(self.image)))

    # def normalize(self, nmin, nmax):
    #     cmin = np.amin(self.image)
    #     cmax = np.amax(self.image)
    #     slope = (nmax-nmin)/(cmax-cmin)
    #
    #     self.image = slope*(self.image-cmin) + nmin
    #     self.print_shape()

    def next_batch(self, batch_size):
        subidx = self.sample_idx(batch_size)
        # self.latents_classes[subidx]

        return self.image[subidx], subidx

