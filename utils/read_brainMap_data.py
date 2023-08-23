from utils.data_tools import *
from utils.local_config import *
import scipy.io as sio
import pickle, time
import pandas as pd
import numpy as np
#import dask.dataframe as dd
#import dask
#import h5py
from json import JSONEncoder
from scipy.io import savemat
import anndata as ad
from scipy.sparse import csr_matrix
#import hdf5plugin
#import pickle5 as pck5

current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def read_smartseq_data(path, saving_path, roi, sub_class, genes, scale=1e6):


    anno_ss = pd.read_csv(path + 'metadata.csv')
    if len(roi) > 0:
        keep_ind = anno_ss['class_label'].isin(sub_class) & anno_ss['region_label'].isin(roi)
        anno_ss = anno_ss[keep_ind].reset_index()
        for r in roi:
            region = 'isocortex_'
    else:
        keep_ind = anno_ss['class_label'].isin(sub_class)
        anno_ss = anno_ss[keep_ind].reset_index()
        region = 'all_'

    anno_ss_size = len(anno_ss.keys())
    print(len(anno_ss))
    chunksize = 10000

    df_ss = pd.DataFrame()
    i = 0
    for chunk_df in pd.read_csv(path + 'matrix.csv', chunksize=chunksize, iterator=True):
        df = pd.merge(anno_ss, chunk_df, how='inner', on='sample_name')
        df_ss = pd.concat([df_ss, df])
        i += 1
        print(i)

    # Normalized UMI counts
    exp_mtx = df_ss.to_numpy()[:, anno_ss_size:]
    cpm = normalize_cellxgene(exp_mtx)
    gene_ss = df_ss.keys()[anno_ss_size:]

    if len(genes) > 0:
        gene_indx = [np.where(gene_ss == gg)[0][0] for gg in genes if np.where(gene_ss == gg)[0]]
    else:
        gene_indx = np.arange(len(gene_ss))

    print(len(gene_indx))

    # Compute the logcpm values for the subset of highly variable genes
    # cpm = np.log1p(scale * cpm[:, gene_indx])
    cpm = np.log1p(scale * cpm)

    # hf = h5py.File(saving_path + 'df_smartseq_' + region + current_time + '.h5', 'w')
    # hf = pd.HDFStore(saving_path + 'test.h5')
    # hf['metadata'] = df_ss.iloc[:, :anno_ss_size]
    # hf['data/log1p'] = cpm
    # hf['data/gene_id'] =  df_ss.keys()[anno_ss_size:]
    # hf['data/count'] = exp_mtx
    # hf.close()
    #
    # g = hf.create_group('metadata')
    # g.create_dataset('metadata', data=df_ss.iloc[:, :anno_ss_size])
    # g = hf.create_group('data/log1p')
    # g.create_dataset('matrix', data=cpm)
    # g = hf.create_group('data/genes')
    # g.create_dataset('gene_id', data=gene_ss)
    # g = hf.create_group('data/count')
    # g.create_dataset('count', data=exp_mtx)
    # print('saving files ....')
    # hf.close()

    dict_ss = {}
    subset_anno = df_ss.iloc[:, :anno_ss_size]
    dict_ss = subset_anno.to_dict('series')
    dict_ss['log1p'] = cpm[:, gene_indx]
    dict_ss['counts'] = exp_mtx[:, gene_indx]
    dict_ss['gene_id'] = gene_ss[gene_indx]
    print(dict_ss['log1p'].shape)

    cluster_label = dict_ss['cluster_label'].values
    cell_type = np.unique(cluster_label)
    all_key = list(dict_ss.keys())
    for key in all_key:
        try:
            dict_ss[key] = dict_ss[key].values
        except:
            dict_ss[key] = dict_ss[key]

    cell_count = []
    for cc in cell_type:
        cell_count.append((cc == cluster_label).sum())

    print(len(cell_count))
    print(sum(cell_count))

    print('saving files ....')
    f = open(saving_path + 'smartseq_' + region + current_time + '.p', "wb")
    pickle.dump(dict_ss, f, protocol=4)
    f.close()
    # with open(saving_path + 'smartseq_' + region + current_time + '.json', 'w') as f:
    #     json.dump(dict_ss, f, cls=NumpyArrayEncoder)
    # f.close()


def read_10x_data(path, saving_path, roi, sub_class, genes=[], scale=1e6):

    anno = pd.read_csv(path + 'metadata.csv')
    if len(roi) > 0:
        keep_ind = anno['class_label'].isin(sub_class) & anno['region_label'].isin(roi)
        anno = anno[keep_ind].reset_index()
        for r in roi:
            region = 'isoCTX_'
    else:
        keep_ind = anno['class_label'].isin(sub_class)
        anno = anno[keep_ind].reset_index()
        region = 'all_'

    anno_size = len(anno.keys())
    print(len(anno))

    h5f_10x = h5py.File(path + 'CTX_Hip_10x_counts.h5', mode='r')
    data_sample_10x = pd.DataFrame({'sample_name': h5f_10x['data/samples'][:].astype(str)})
    data_sample_10x['sample_name'] = np.array([s[10:1+np.char.find(s, '-')] + s[np.char.find(s, 'L8TX'):] for s in data_sample_10x['sample_name']])
    anno_10x_ord = pd.merge(anno.reset_index(), data_sample_10x.reset_index(), how='inner', on='sample_name')
    gene_10x = h5f_10x['data/gene'][()].astype(str)

    print('read count matrix')
    dict_10x = {}
    subset_anno = anno_10x_ord.iloc[:, :anno_size]
    dict_10x = subset_anno.to_dict('series')
    exp_mtx = h5f_10x['data/counts'][()][:, anno_10x_ord.index_y].T
    # exp_mtx = exp_mtx[gene_indx, :]
    # anno_10x_counts = anno_10x_ord['cluster_label'].value_counts().to_frame()
    # anno_10x_counts.reset_index(inplace=True)
    # anno_10x_counts.columns = ['cluster_label', '10x_count']

    print(np.unique(dict_10x['region_label']))
    chunksize = 20000
    #
    # df_ss = pd.DataFrame()
    # i = 0
    # for chunk_df in pd.read_csv(path + 'matrix.csv', chunksize=chunksize, iterator=True):
    #     df = pd.merge(anno, chunk_df, how='inner', on='sample_name')
    #     df_ss = pd.concat([df_ss, df])
    #     i += 1
    #     print(i)

    # Normalized UMI counts
    #exp_mtx = df_ss.to_numpy()[:, anno_size:]

    dict_10x['log1p'] = np.zeros((exp_mtx.shape[0], len(gene_10x)))

    print('cpm calculation')
    for iter in range(exp_mtx.shape[0] // chunksize + 1):
        ind_0 = iter * chunksize
        ind_1 = np.min((exp_mtx.shape[0], (iter + 1) * chunksize))
        print(ind_0, ind_1)
        tmp = normalize_cellxgene(exp_mtx[ind_0:ind_1, :])
        dict_10x['log1p'][ind_0:ind_1, :] = np.log1p(scale * tmp)
        print(iter)

    print(np.sum(dict_10x['log1p'], axis=1))
    # gene_ss = df_ss.keys()[anno_size:]
    #
    # if len(genes) > 0:
    #     gene_indx = [np.where(gene_ss == gg)[0][0] for gg in genes if np.where(gene_ss == gg)[0]]
    # else:
    #     gene_indx = np.arange(len(gene_ss))
    #
    # print(len(gene_indx))
    #
    # Compute the logcpm values for the subset of highly variable genes
    # cpm = np.log1p(scale * cpm[:, gene_indx])
    # cpm = np.log1p(scale * cpm)

    #
    # dict_10x['log1p'] = cpm
    # dict_10x['counts'] = dict_10x['counts'][:, gene_indx]
    dict_10x['gene_id'] = gene_10x #[g_ind[:n_gene]]
    if len(genes) > 0:
        gene_idx = [np.where(dict_10x['gene_id'])[0] for gg in genes]
        gene_idx = np.concatenate(gene_idx).astype(int)
        dict_10x['gene_id'] = dict_10x['gene_id'][gene_idx]
        dict_10x['log1p'] = dict_10x['log1p'][:, gene_idx]

        x = csr_matrix(dict_10x['log1p'])
        var = pd.DataFrame(dict_10x['gene_id'], columns=['features'])

        delete = ['gene_id', 'log1p']
        for key in delete:
            del dict_10x[key]

        anno = pd.DataFrame.from_dict(dict_10x)
        adata = ad.AnnData(X=x, obs=anno, var=var, dtype='float32')
        file_name = saving_path + sub_class[0] + '_' + region + '_nGene_' + str(len(gene_idx)) + '_' + current_time + '.h5ad'
        adata.write_h5ad(file_name)
        
    else:
        print(dict_10x['log1p'].shape)

        cluster_label = dict_10x['cluster_label'].values
        cell_type = np.unique(cluster_label)
        all_key = list(dict_10x.keys())
        for key in all_key:
            try:
               dict_10x[key] = dict_10x[key].values
            except:
                dict_10x[key] = dict_10x[key]

        cell_count = []
        for cc in cell_type:
            cell_count.append((cc == cluster_label).sum())

        print(len(cell_count))
        print(sum(cell_count))

        file_name = saving_path + sub_class[0] + 'data_' + region + current_time + '.p'
        f = open(file_name, 'wb')
        pickle.dump(dict_10x, f, protocol=5)
        f.close()

    return file_name



def get_hvg(file_path, file_name, n_gene, saving_path, keep_keys, chunksize=10000):

    print('loading 10x file ...' + file_path)
    #data = pck5.load(open(file_path, 'rb'))
    data = pickle.load(open(file_path, 'rb'))
    print('data is loaded.')
    # print((np.sum(data['log1p'], axis=1)))
    t_gene = len(data['gene_id'])
    print(t_gene)
    print(int(t_gene // chunksize))
    g_std = []
    for iter in range(int(t_gene //chunksize) + 1):
        ind0 = iter * chunksize
        ind1 = np.min((t_gene, (iter + 1)*chunksize))
        g_std.append(np.std(data['log1p'][:, ind0:ind1], axis=0))
    
    g_std = np.concatenate((g_std))
    print(len(g_std))
    g_ind = np.argsort(g_std)
    g_ind = g_ind[::-1]
    x = csr_matrix(data['log1p'][:, g_ind[:n_gene]])
    var = pd.DataFrame(data['gene_id'][g_ind[:n_gene]], columns=['features'])
    
    delete = [key for key in data if key not in keep_keys]
    for key in delete:
        del data[key]

    #data.pop('gene_id', None)
    #data.pop('gene_id', None)
    anno = pd.DataFrame.from_dict(data)
    adata = ad.AnnData(X=x, obs=anno, var=var, dtype='float32')

    filename = saving_path + file_name + '_nGene_' + str(n_gene) + '.h5ad'
    adata.write_h5ad(filename) #, compression=hdf5plugin.FILTERS["zstd"])

    # try:
    #     file_name = saving_path + file_name + '_nGene_' + str(n_gene) + '.mat'
    #     savemat(file_name, data, do_compression=True)
    # except:
    #     f = open(file_name, 'wb')
    #     pickle.dump(data, f, protocol=5)
    #     f.close()



