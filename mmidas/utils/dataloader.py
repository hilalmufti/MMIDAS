import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.special import softmax
import scanpy
import anndata
import hdf5plugin

def get_genes(gene_id, n_genes):

    gaba_ind_1, gaba_ind_2, glutam_ind = np.array([]), np.array([]), np.array([])
    glutam_genes = ['Slc30a3', 'Cux2', 'Rorb', 'Deptor', 'Scnn1a', 'Rspo1',
                    'Hsd11b1', 'Batf3', 'Oprk1', 'Osr1', 'Car3', 'Fam84b',
                    'Chrna6', 'Pvalb', 'Pappa2', 'Foxp2', 'Slc17a8', 'Trhr',
                    'Tshz2', 'Rapdegf3', 'Trh', 'Gpr139', 'Nxph4', 'Rprm',
                    'Crym', 'Nxph3', 'Nlgn1', 'C1ql2', 'C1ql3', 'Adgrl1', 'Nlgn3',
                    'Dag1', 'Cbln1', 'Lrrtm1']

    gaba_genes_1 = ['Lamp5', 'Ndnf', 'Krt73', 'Fam19a1', 'Pax6', 'Ntn1', 'Plch2',
                    'Lsp1', 'Lhx6', 'Nkx2.1', 'Vip', 'Sncg', 'Slc17a8', 'Nptx2',
                    'Gpr50', 'Itih5', 'Serpinf1', 'Igfbp6', 'Gpc3', 'Lmo1',
                    'Ptprt', 'Rspo4', 'Chat', 'Crispld2', 'Col15a1', 'Pde1a',
                    'Cbln2', 'Cbln4', 'C1ql1', 'Lrrtm3', 'Clstn3', 'Nlgn2',
                    'Nr2e1', 'Unc5a', 'Rgs16', 'Kcnh3', 'Celsr3']

    gaba_genes_2 = ['Sst', 'Chodl', 'Nos1', 'Mme', 'Tac1', 'Tacr3', 'Calb2',
                    'Nr2f2', 'Myh8', 'Tac2', 'Hpse', 'Crchr2', 'Crh', 'Esm1',
                    'Rxfp1', 'Nts', 'Pvalb', 'Gabrg1', 'Th', 'Calb1',
                    'Akr1c18', 'Sea3e', 'Gpr149', 'Reln', 'Tpbg', 'Cpne5',
                    'Vipr2', 'Nkx2-1', 'Lrrtm3', 'Clstn3', 'Nlgn2', 'Cbln3',
                    'Lrrtm2', 'Nxph1', 'Nxph2', 'Nxph4', 'Syt2', 'Hapln4',
                    'St6galnac5', 'Etv6', 'Iqgap2', 'Rasgef1b', 'Oxtr', 'Lama4',
                    'Lipa', 'Sirt4']

    for g in glutam_genes:
        glutam_ind = np.append(glutam_ind, np.array([i for i, item in enumerate(gene_id) if g == item]))

    glutam_gene_ind = list(map(int, glutam_ind))

    for g in gaba_genes_1:
        gaba_ind_1 = np.append(gaba_ind_1, np.array([i for i, item in enumerate(gene_id) if g == item]))
    gaba_gene_ind_1 = list(map(int, gaba_ind_1))

    for g in gaba_genes_2:
        gaba_ind_2 = np.append(gaba_ind_2, np.array([i for i, item in enumerate(gene_id) if g == item]))
    gaba_gene_ind_2 = list(map(int, gaba_ind_2))

    gene_indx = np.concatenate((glutam_gene_ind, gaba_gene_ind_1, gaba_gene_ind_2))
    if n_genes > 0:
        gene_index = np.unique(np.concatenate((np.array(range(n_genes)), gene_indx)))
    else:
        gene_index = np.unique(np.concatenate((np.array(range(len(gene_id))), gene_indx)))

    return gene_index


def get_10x_clusters(file_smt, file_10x, min_ratio=5, eps=1e-1):

    ann_smt = pd.read_csv(file_smt)
    ann_10x = pd.read_csv(file_10x)

    keep_ind = ~ann_smt['cluster_label'].isnull()
    ann_smt = ann_smt[keep_ind].reset_index()

    keep_ind = ~ann_10x['cluster_label'].isnull()
    ann_10x = ann_10x[keep_ind].reset_index()

    smart_types = ann_smt['cluster_label'].astype(str)
    x10_types = ann_10x['cluster_label'].astype(str)

    clusters = np.unique(np.concatenate((np.unique(smart_types), np.unique(x10_types))))
    x10_num = np.array([sum(x10_types == cc) for cc in clusters])
    smart_num = np.array([sum(smart_types == cc) for cc in clusters])
    ratio = x10_num / (smart_num + eps)

    return clusters[ratio > min_ratio]



def load_data(datafile, n_gene=0, gene_id=[], ref_genes=False, ref_types=False, ann_smt='', ann_10x='', eps=1e-6, tau=0.11, min_num=10):

    key_list = ['log1p', 'gene_id', 'sample_name', 'cluster_color', 'cluster_order', 'cluster_label', 'class_color',
                'class_order', 'class_label', 'subclass_color', 'subclass_order', 'subclass_label', 'sex_id',
                'donor_sex_label', 'region_color', 'region_id', 'region_label', 'cortical_layer_label']
    adata = anndata.read_h5ad(datafile)
    print('data is loaded.')

    data = dict()
    data['log1p'] = adata.X
    features = adata.var.keys()
    data['gene_id'] = adata.var[features].values
    anno_key = adata.obs.keys()
    for key in anno_key:
        data[key] = adata.obs[key].values

    if n_gene == 0:
        n_gene = len(data['gene_id'])

    if len(gene_id) > 0:
        gene_idx = [np.where(data['gene_id'] == gg)[0] for gg in gene_id]
        gene_idx = np.concatenate(gene_idx).astype(int)
        data['gene_id'] = data['gene_id'][gene_idx]
        data['log1p'] = data['log1p'][:, gene_idx].todense()
        n_gene = len(gene_idx)
    else:
        data['gene_id'] = data['gene_id'][:n_gene]
        data['log1p'] = data['log1p'][:, :n_gene].todense()

    if ref_genes:
        gene_idx = get_genes(data['gene_id'], n_gene)
        data['log1p'] = data['log1p'][:, gene_idx]
        data['gene_id'] = data['gene_id'][gene_idx]

    print(data['log1p'].shape)

    all_key = list(data.keys())
    for key in all_key:
        if key not in key_list:
            del data[key]

    # remove cells "CR", "NP PPP", "Ndnf", and "Pax"
    ind = np.where(data['cluster_label'] == '1_CR')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    ind = np.where(data['cluster_label'] == '354_NP PPP')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    ind = np.where(data['cluster_label'] == '22_Ndnf HPF')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    ind = np.where(data['cluster_label'] == '23_Ndnf HPF')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    ind = np.where(data['cluster_label'] == '17_Pax6')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    ind = np.where(data['cluster_label'] == '18_Pax6')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    ind = np.where(data['cluster_label'] == '19_Pax6')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    # ref_len = len(data['cluster_label'])
    # all_key = list(data.keys())
    # for key in all_key:
    #     if len(data[key]) >= ref_len:
    #         data[key] = np.delete(data[key], ind, axis=0)

    uniq_clusters = np.unique(data['cluster_label'])

    if ref_types:
        ref_10x = get_10x_clusters(ann_smt, ann_10x)
        n_types = np.array([sum(ref_10x == cc) for cc in uniq_clusters])
        for rmv_type in uniq_clusters[n_types == 0]:
            ind = np.where(data['cluster_label'] == rmv_type)[0]
            for key in data:
                if key not in ['gene_id']:
                    data[key] = np.delete(data[key], ind, axis=0)
        uniq_clusters = np.unique(data['cluster_label'])

    count = np.zeros(len(uniq_clusters))
    for it, tt in enumerate(uniq_clusters):
        count[it] = sum(data['cluster_label'] == tt)

    uniq_clusters = uniq_clusters[count >= min_num]
    subclass_ind = []
    for tt in uniq_clusters:
        subclass_ind.append(np.array([i for i in range(len(data['cluster_label'])) if tt == data['cluster_label'][i]]))

    subclass_ind = np.concatenate(subclass_ind)
    ref_len = len(data['cluster_label'])

    for k in key_list:
        if k == 'log1p':
            data[k] = np.array(data[k])[subclass_ind, :]
        elif k not in ['gene_id', 'log1p']:
            data[k] = np.array(data[k])[subclass_ind]

    data['cluster_id'] = np.zeros(len(data['cluster_order']))
    for ic, cls_ord in enumerate(np.unique(data['cluster_label'])):
        data['cluster_id'][data['cluster_label'] == cls_ord] = int(ic + 1)

    # label_encoder = LabelEncoder()
    # integer_encoded = label_encoder.fit_transform(data['cluster_order'])
    # onehot_encoder = OneHotEncoder(sparse=False)
    # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # data['c_onehot'] = onehot_encoder.fit_transform(integer_encoded)
    # data['c_p'] = softmax((data['c_onehot'] + eps) / tau, axis=1)
    #
    # data['n_type'] = len(np.unique(data['cluster_order']))

    print(len(np.unique(data['cluster_label'])), data['log1p'].shape, len(data['gene_id']))

    return data




