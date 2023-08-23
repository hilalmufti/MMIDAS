import os
from utils.training import train_cplmixVAE
from utils.training_VAE import train_VAE
import pickle
from utils.state_analysis import state_analyzer
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import random
from utils.dataloader import load_data
from utils.config import load_config
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.special import softmax
import seaborn as sns
import glob
import numpy as np
from utils.data_tools import reorder_genes



def eval_mixmodel(data,
                   n_categories,
                   n_arm,
                   state_dim,
                   latent_dim=30,
                   latent_dim_ae=2,
                   fc_dim=100,
                   temp=1,
                   saving_folder='',
                   ae_folder='',
                   ae_model='',
                   device=None,
                   hard=False,
                   tau=0.01,
                   n_gene=0,
                   n_zim=1,
                   batch_size=5000,
                   models=[],
                   loss_mode='MSE',
                   subclass='',
                   sort_gene=True,
                   variational=True,
                   ref_pc=False,
                   save=False):


    paths = load_config(config_file='config.toml')
    if len(data) == 0:
        if subclass:
            data_file = paths['local_data_path'] / paths['data_' + subclass]
            data = load_data(datafile=data_file)  # , ref_types=True, ann_smt=paths['ann_smt'], ann_10x=paths['ann_10x'])

            if sort_gene:
                g_index = reorder_genes(data['log1p'])
                # if n_gene == 0:
                #     n_gene = len(g_index) // n_arm

                # g_idx = g_index[0::n_arm]
                # for arm in range(1, n_arm):
                #     g_idx = np.concatenate((g_idx, g_index[arm::n_arm]))

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

    uniq_cluster_order = np.unique(data['cluster_order'])


    cpl_mixVAE = train_cplmixVAE(saving_folder=saving_folder,
                                 device=device,
                                 n_feature=n_gene)

    train_loader, test_loader, alldata_loader, train_ind, test_ind = cpl_mixVAE.getdata(dataset=data['log1p'],
                                                                                        label=data['cluster_order'],
                                                                                        batch_size=batch_size)
    data_indx = np.concatenate((train_ind, test_ind))

    cpl_mixVAE.init_model(n_categories=n_categories,
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

    if ae_folder:
        ae = train_VAE(saving_folder=ae_folder, device=device)

        ae.init_model(input_dim=data['log1p'].shape[1],
                      fc_dim=fc_dim,
                      lowD_dim=latent_dim_ae,
                      x_drop=0.,
                      lr=0.,
                      variational=False)

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

    data['cluster_id'] = data['cluster_id'][data_indx]

    for i in range(len(models)):
        print(models[i])
        cpl_mixVAE.load_model(models[i])
        if ref_pc:
            # data_file_id = cpl_mixVAE.eval_model(data_loader=alldata_loader, g_idx=g_idx, gc_p=data['c_p'], c_onehot=data['c_onehot'], batch_size=batch_size)
            outcome, data_file_id = cpl_mixVAE.eval_model(data_loader=alldata_loader, g_idx=g_idx, gc_p=data['c_p'], c_onehot=data['c_onehot'], batch_size=batch_size)
        else:
            # data_file_id = cpl_mixVAE.eval_model(data_loader=alldata_loader, g_idx=g_idx, batch_size=batch_size)
            outcome, data_file_id = cpl_mixVAE.eval_model(data_mat=data['log1p'], batch_size=batch_size, mode=loss_mode)

        if ae_folder:
            ae.load_model(ae_model)
            ae_file = ae.eval_model(data_mat=data['log1p'])
            ae_out = ae.load_file(ae_file)
            z_ae = ae_out['z']
            data_rec = ae_out['x_recon']

        # outcome = cpl_mixVAE.load_file(data_file_id)
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
                # armA_vs_armB_norm = armA_vs_armB_norm[:, ind_sort[::-1]][ind_sort[::-1]]
                # armA_vs_armB = armA_vs_armB[:, ind_sort[::-1]][ind_sort[::-1]]
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

    return cpl_mixVAE, data, data_dic





