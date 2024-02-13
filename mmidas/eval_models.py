import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.special import softmax
from .data_tools import reorder_genes
from .cpl_mixvae import cpl_mixVAE
from .dataloader import load_data
from .config import load_config


def eval_mixmodel(data,
                   n_categories,
                   n_arm,
                   state_dim,
                   latent_dim=30,
                   fc_dim=100,
                   temp=1,
                   saving_folder='',
                   device=None,
                   hard=False,
                   tau=0.01,
                   n_gene=0,
                   batch_size=5000,
                   models=[],
                   loss_mode='MSE',
                   subclass='',
                   sort_gene=True,
                   variational=True,
                   ref_pc=False,
                   save=False):
    
    """
        Initialized the deep mixture model and its optimizer.

        input args:
            data: a dictionary that contains the following, 'log1p' for the gene expression matrix, 
                  'cluster_id' for the cluster label, and 'cluster_label' for the cluster name, 'gene_id' for the gene names.
            n_categories: number of categories in the categorical variable.
            n_arm: int value that indicates number of arms.
            state_dim: dimension of the state variable.
            latent_dim: dimension of the latent representation.
            fc_dim: dimension of the hidden layer.
            temp: temperature of sampling.
            saving_folder: a string that indicates the folder to save the model(s) and file(s).
            device: device: computing device, either 'cpu' or 'cuda'. For 'cpu' mode, the device is None. For 'cuda' mode, the device is an integer.
            hard: a boolean variable, True uses one-hot method that is used in Gumbel-softmax, and False uses the Gumbel-softmax function.
            tau: temperature of the Gumbel-softmax function.
            n_gene: number of genes to be used in the analysis.
            batch_size: batch size for dataloader.
            models: a list of pre-trained models.
            loss_mode: the loss function, either 'MSE' or 'ZINB'.
            subclass: a string that indicates the subclass of the data, e.g. 'Glutamatergic', 'GABAergic', or 'Sst'.
            sort_gene: a boolean variable, True sorts the genes based on the variance, False does not sort the genes.
            variational: a boolean variable for variational mode, False mode does not use sampling.
            ref_pc: a boolean variable, True uses the reference prior for the categorical variable.
            save: a boolean variable, True saves the model and the results, False does not save the model and the results.
    """


    paths = load_config(config_file='config.toml')
    if len(data) == 0:
        if subclass:
            data_file = paths['local_data_path'] / paths['data_' + subclass]
            data = load_data(datafile=data_file)  

            if sort_gene:
                g_index = reorder_genes(data['log1p'])
            

            data['log1p'] = data['log1p'][:, g_index[:n_gene]]
            data['gene_id'] = data['gene_id'][g_index[:n_gene]]

        else:
            data_file_glum = paths['local_data_path'] / paths['data_glum']
            data_file_gaba = paths['local_data_path'] / paths['data_gaba']
            data_gaba = load_data(data_file_gaba)
            data_glum = load_data(data_file_glum)

            data = dict()
            data['gene_id'] = data_gaba['gene_id']
            for key in data_glum.keys():
                if key != 'gene_id':
                    data[key] = np.concatenate((data_glum[key], data_gaba[key]))

    eps = 1e-6
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data['cluster_id'])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    data['c_onehot'] = onehot_encoder.fit_transform(integer_encoded)
    data['c_p'] = softmax((data['c_onehot'] + eps) / tau, axis=1)
    data['n_type'] = len(np.unique(data['cluster_id']))

    print('done')
    print(data['log1p'].shape, len(np.unique(data['cluster_label'])), n_gene)

    if n_gene == 0:
        n_gene = data['log1p'].shape[1]

    cplMixVAE = cpl_mixVAE(saving_folder=saving_folder,
                                 device=device,
                                 n_feature=n_gene)

    cplMixVAE.init_model(n_categories=n_categories,
                          state_dim=state_dim,
                          input_dim=n_gene,
                          fc_dim=fc_dim,
                          lowD_dim=latent_dim,
                          x_drop=0.,
                          s_drop=0.,
                          lr=0.,
                          n_arm=n_arm,
                          temp=temp,
                          hard=hard,
                          tau=tau,
                          lam=1,
                          lam_pc=1,
                          beta=1,
                          n_zim=n_zim,
                          variational=variational,
                          ref_prior=ref_pc)

    recon_loss = []
    label_pred = []
    test_dist_z = []
    test_dist_qz = []
    n_pruned = []
    consensus_min = []
    consensus_mean = []
    cT_vs_cT = []
    test_loss = [[] for arm in range(n_arm)]
    prune_indx = []
    consensus = []
    AvsB = []
    sample_id = []
    z_ae = []
    data_rec = []


    for i in range(len(models)):
        print(models[i])
        cplMixVAE.load_model(models[i])
        if ref_pc:
            outcome = cplMixVAE.eval_model(data_mat=data['log1p'], c_p=data['c_p'], batch_size=batch_size, mode=loss_mode)
        else:
            outcome = cplMixVAE.eval_model(data_mat=data['log1p'], batch_size=batch_size, mode=loss_mode)

        x_low = outcome['x_low']
        predicted_label = outcome['predicted_label']
        test_dist_z.append(outcome['total_dist_z'])
        test_dist_qz.append(outcome['total_dist_qz'])
        recon_loss.append(outcome['total_loss_rec'])
        c_prob = outcome['z_prob']
        c_sample = outcome['z_sample']
        prune_indx.append(outcome['prune_indx'])
        sample_id.append(outcome['data_indx'])
        label_pred.append(predicted_label)

        category_vs_class = np.zeros((n_arm, data['n_type'], n_categories))

        for arm in range(n_arm):
            test_loss[arm].append(outcome['total_loss_rec'][arm])
            label_predict = []
            for d in range(len(data['cluster_id'])):
                z_cat = np.squeeze(c_prob[arm][d, :])
                category_vs_class[arm, int(data['cluster_id'][d] - 1), np.argmax(z_cat)] += 1

        if ref_pc:
            n_arm += 1

        for arm_a in range(n_arm):
            pred_a = predicted_label[arm_a, :]
            for arm_b in range(arm_a + 1, n_arm):
                pred_b = predicted_label[arm_b, :]
                armA_vs_armB = np.zeros((n_categories, n_categories))

                for samp in range(pred_a.shape[0]):
                    armA_vs_armB[np.int(pred_a[samp]) - 1, np.int(pred_b[samp]) - 1] += 1

                num_samp_arm = []
                for ij in range(n_categories):
                    sum_row = armA_vs_armB[ij, :].sum()
                    sum_column = armA_vs_armB[:, ij].sum()
                    num_samp_arm.append(max(sum_row, sum_column))

                armA_vs_armB_norm = np.divide(armA_vs_armB, np.array(num_samp_arm), out=np.zeros_like(armA_vs_armB),
                                         where=np.array(num_samp_arm) != 0)
                nprune_indx = np.where(np.isin(range(n_categories), prune_indx[i]) == False)[0]
                armA_vs_armB_norm = armA_vs_armB_norm[:, nprune_indx][nprune_indx]
                armA_vs_armB = armA_vs_armB[:, nprune_indx][nprune_indx]
                diag_term = np.diag(armA_vs_armB_norm)
                ind_sort = np.argsort(diag_term)
                consensus_min.append(np.min(diag_term))
                con_mean = 1. - (sum(np.abs(predicted_label[0, :] - predicted_label[1, :]) > 0.) / predicted_label.shape[1])
                consensus_mean.append(con_mean)
                AvsB.append(armA_vs_armB)
                consensus.append(armA_vs_armB_norm)

        n_pruned.append(len(nprune_indx))
        category_vs_class = category_vs_class[:, :, nprune_indx]
        cT_vs_cT.append(category_vs_class)
        plt.close()

    data_dic = {}
    data_dic['recon_loss'] = test_loss
    data_dic['dz'] = test_dist_z
    data_dic['d_qz'] = test_dist_qz
    data_dic['con_min'] = consensus_min
    data_dic['con_mean'] = consensus_mean
    data_dic['num_pruned'] = n_pruned
    data_dic['pred_label'] = label_pred
    data_dic['cT_vs_cT'] = cT_vs_cT
    data_dic['consensus'] = consensus
    data_dic['armA_vs_armB'] = AvsB
    data_dic['prune_indx'] = prune_indx
    data_dic['nprune_indx'] = nprune_indx
    data_dic['state_mu'] = outcome['state_mu']
    data_dic['state_sample'] = outcome['state_sample']
    data_dic['state_var'] = outcome['state_var']
    data_dic['sample_id'] = sample_id
    data_dic['c_prob'] = c_prob
    data_dic['lowD_x'] = x_low
    data_dic['z_ae'] = z_ae
    data_dic['x_rec'] = data_rec

    if save:
        f_name = saving_folder + '/summary_performance_K_' + str(n_categories) + '_narm_' + str(n_arm) + '.p'
        f = open(f_name, "wb")
        pickle.dump(data_dic, f)
        f.close()

    return cplMixVAE, data, data_dic





