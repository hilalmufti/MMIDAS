import os
from utils.training_allgene import train_cplmixVAE
from utils.training_VAE import train_VAE
import pickle
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from utils.data_tools import reorder_genes
import seaborn as sns
import glob
from sklearn.model_selection import train_test_split
import numpy as np
from utils.dataloader import load_data


def eval_mixmodel(n_categories,
               n_arm,
               state_dim,
               latent_dim=30,
               latent_dim_ae=2,
               fc_dim=100,
               temp=1,
               saving_folder='',
               ae_folder='',
               ae_model='',
               data_file='',
               device=None,
               hard=False,
               tau=0.01,
               n_gene=0,
               models=[],
               all_data=False,
               ref_pc=False,
               save=False):

    # load Patch-seq data

    data = load_data(file=data_file, n_gene=n_gene, ref_genes=True)
    overlap = 0.5
    g_index = reorder_genes(data['log1p'])
    if n_gene == 0:
        n_gene = len(g_index) // n_arm

    g_overlap = int(n_gene // (1 / overlap))
    g_idx = np.concatenate((g_index[0:g_overlap], g_index[g_overlap::n_arm]))
    for arm in range(1, n_arm):
        g_set = np.concatenate((g_index[0:g_overlap], g_index[g_overlap + arm::n_arm]))
        g_idx = np.concatenate((g_idx, g_set))

    data['log1p'] = data['log1p'][:, g_idx]
    data['gene_id'] = data['gene_id'][g_idx]
    n_gene = data['log1p'].shape[1] // n_arm

    print(data['log1p'].shape, len(np.unique(data['cluster'])), n_gene)

    cpl_mixVAE = train_cplmixVAE(saving_folder=saving_folder,
                                 device=device,
                                 n_feature=n_gene)

    train_loader, test_loader, alldata_loader, train_ind, test_sample = cpl_mixVAE.getdata(dataset=data['log1p'], label=data['cluster'], batch_size=1000)

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
                          lam=1.,
                          lam_pc=1.,
                          beta=1.,
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

    for i in range(len(models)):
        # print(models[i])
        cpl_mixVAE.load_model(models[i])
        if all_data:
            x = data['log1p']
            cluster_id = data['clusterID']
        else:
            x = data['log1p'][test_sample, :]
            cluster_id = data['clusterID'][test_sample]

        if ref_pc:
            data_file_id = cpl_mixVAE.eval_model(x, c_p=data['c_p'], c_onehot=data['c_onehot'])
        else:
            data_file_id = cpl_mixVAE.eval_model(x)

        if ae_folder:
            ae.load_model(ae_model)
            ae_file = ae.eval_model(data_mat=x)
            ae_out = ae.load_file(ae_file)
            z_ae = ae_out['z']
            data_rec = ae_out['x_recon']

        outcome = cpl_mixVAE.load_file(data_file_id)
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

        category_vs_class = np.zeros((n_arm, n_categories, n_categories))

        for arm in range(n_arm):
            test_loss[arm].append(outcome['total_loss_rec'][arm])

            label_predict = []
            for d in range(len(cluster_id)):
                z_cat = np.squeeze(c_prob[arm][d, :])
                category_vs_class[arm, int(cluster_id[d] - 1), np.argmax(z_cat)] += 1
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
                # armA_vs_armB_norm = armA_vs_armB_norm[:, nprune_indx][nprune_indx]
                n_pruned.append(len(nprune_indx))
                diag_term = np.diag(armA_vs_armB_norm)
                ind_sort = np.argsort(diag_term)
                armA_vs_armB_norm = armA_vs_armB_norm[:, ind_sort[::-1]][ind_sort[::-1]]
                consensus_min.append(np.min(diag_term))
                con_mean = 1. - (sum(np.abs(predicted_label[0, :] - predicted_label[1, :]) > 0.) / predicted_label.shape[1])
                consensus_mean.append(con_mean)
                AvsB.append(armA_vs_armB)
                consensus.append(armA_vs_armB_norm)

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
    data_dic['z'] = x_low
    data_dic['z_ae'] = z_ae
    data_dic['x_rec'] = data_rec

    if save:
        f_name = saving_folder + '/summary_performance_K_' + str(n_categories) + '_narm_' + str(n_arm) + '.p'
        f = open(f_name, "wb")
        pickle.dump(data_dic, f)
        f.close()

    return cpl_mixVAE, data, data_dic





