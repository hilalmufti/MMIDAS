from utils.training import train_cplmixVAE
from utils.training_VAE import train_VAE
import pickle
from utils.config import load_config
from utils.eval_models import eval_mixmodel
from utils.gene_analysis import traversal_analysis, get_gene_indx, kegg_genes, genes_subset
import matplotlib.pyplot as plt
from numpy.linalg import norm
import seaborn as sns
import glob, os
import numpy as np
import torch
import toml
import pandas as pd
from utils.dataloader import load_data
from scipy.spatial import distance


n_run = 1
temp = 1.
n_categories = 120
state_dim = 2
n_arm = 2
tau = .005
beta = 1
fc_dim = 100
latent_dim = 10
lr = 0.001
p_drop = .5
n_gene = 5023
batch_size = 5000
n_epoch = 10000
n_epoch_p = 1000
n_gene = 5000
device = None
augmentation = True
ref_prior = False
model_order = 92
shift = 3

# singlaing_pathway = ["Circadian rhythm", "Circadian entrainment", "Oxidative", "Wnt signaling", "Ca signaling",
#                      "cAMP signaling", "RNA degradation", "cell adhesion"]

paths = load_config(config_file='config.toml')
saving_folder = paths['package_dir'] / paths['saving_folder_cplmix']
data_path = paths['package_dir'] / paths['data']
folder_name = f'run_{n_run}_K_{n_categories}_Sdim_{state_dim}_p_drop_{p_drop}_fc_dim_{fc_dim}_Prior_{ref_prior}_aug_{augmentation}' + \
    f'_lr_{lr}_n_arm_{n_arm}_tau_{tau}_temp_1_beta_{beta}_nbatch_{batch_size}_nepoch_{n_epoch}_nepochP_{n_epoch_p}'

n_comb = max(n_arm * (n_arm - 1) / 2, 1)
saving_folder = str(saving_folder / folder_name)

selected_model = glob.glob(saving_folder + '/model/cpl_mixVAE_model_after_pruning_' + str(n_categories - model_order) + '*')[0]
cpl_mixVAE, data, outcome = eval_mixmodel(n_categories=n_categories,
                                       state_dim=state_dim,
                                       n_arm=n_arm,
                                       latent_dim=latent_dim,
                                       fc_dim=fc_dim,
                                       tau=tau,
                                       n_gene=n_gene,
                                       saving_folder=saving_folder,
                                       data_file=data_path,
                                       ref_pc=ref_prior,
                                       all_data=True,
                                       device=device,
                                       models=[selected_model])


c_label = outcome['pred_label'][-1][0, :]
mismatch = outcome['pred_label'][-1][0, :] != outcome['pred_label'][-1][1, :]
cp_mean = np.mean((outcome['c_prob'][0, mismatch, :], outcome['c_prob'][1, mismatch, :]), axis=0)
c_label[mismatch] = np.argmax(cp_mean, axis=1) + 1
c_cat = np.unique(c_label)
pruning_mask = outcome['nprune_indx']
s_idx = outcome['sample_id'][-1].astype(int)

gene_exp = data['log1p'][s_idx, :]
t_type = data['cluster'][s_idx]
n_gene = len(data['gene_id'])

ieg, hkg, NPP_genes, NPGPCR_genes = genes_subset()

g_subset = []
g_subset.append(get_gene_indx(ref_gene=data['gene_id'], traget_gene=NPP_genes))
g_subset.append(get_gene_indx(ref_gene=data['gene_id'], traget_gene=NPGPCR_genes))
# g_subset.append(get_gene_indx(ref_gene=data['gene_id'], traget_gene=ieg))
# g_subset.append(get_gene_indx(ref_gene=data['gene_id'], traget_gene=hkg))

kegg_pathway_genes = toml.load(paths['package_dir'] / 'KEGG.toml')
kegg_pathway = pd.read_csv(paths['package_dir'] / 'KEGG_pathways.csv')

g_id_set = []
g_id_set.append(NPP_genes)
g_id_set.append(NPGPCR_genes)
for pathway in kegg_pathway.pathway_id:
    gg = kegg_pathway_genes[pathway]['facs_genes']
    g_id_set.append(gg)
    g_subset.append(get_gene_indx(ref_gene=data['gene_id'], traget_gene=gg))

signaling_pathways = np.concatenate((['NPP', 'NP-GPCR'], kegg_pathway.pathway_name.values))
g_idx = np.concatenate(g_subset)
# g_subset = [NPPg_idx, NPGPCRg_idx, ieg_idx, hkg_idx]
selected_genes = data['gene_id'][g_idx]

min_n_cell = 1000
selected_c = [80, 119, 1, 13, 92, 25, 55, 110, 62, 69, 31]
color = ['#DDACC9', '#DD6091', '#DD6091', '#804811', '#9189FF', '#FF00FF', '#FF00B3', '#008F39', '#D9F077', '#7AE6AB', '#0094C2']
c_ano = ['Lamp5 T_45', 'Lamp5 T_46', 'Lamp5 T_47', 'Sst_73', 'Vip T_61', 'Vip T_64', 'Vip T_65', 'L5 NP T_32', 'L2/3 IT T_1',
        'L2/3 IT T_4', 'L2/3 IT T_14']

g_std = np.std(data['log1p'], axis=0)
x = np.arange(len(g_idx))


corr_mean = np.zeros((model_order, state_dim, n_gene, n_arm))
corr_std = np.zeros((model_order, state_dim, n_gene, n_arm))
p_val_mean = np.zeros((model_order, state_dim, n_gene, n_arm))
corr = np.zeros((model_order, state_dim, min_n_cell, n_gene, n_arm))
g_var_mean = np.zeros((model_order, n_arm, state_dim, n_gene))
g_var_std = np.zeros((model_order, n_arm, state_dim, n_gene))
n_cell = np.zeros(model_order)
ref_type = []
med_c_gene = np.zeros((model_order, n_gene))
mode = 'pca'
s_mu, s_travers = [], []

for i_cat, cat in enumerate(c_cat): #enumerate(c_cat)

    indx = np.where(c_label == cat)[0]
    print(i_cat, cat, len(indx))
    ref_type.append(np.unique(t_type[indx]))
    med_c_gene[i_cat, :] = np.median(gene_exp[indx, :], axis=0)
    recon_genes_s, recon_m, state_s,  state_mu = traversal_analysis(gene_exp[indx, :], model=cpl_mixVAE, state_dim=state_dim, mask=pruning_mask, temp=temp, mode=mode)
    sum_exp_c = np.sum(gene_exp[indx, :], axis=0)

    if mode == 'pca':
        mu = state_mu.cpu().detach().numpy()
        smp = state_s.cpu().detach().numpy()
        s_mu.append(mu)
        s_travers.append(smp)
        exp_mu = recon_m.cpu().detach().numpy()
        traversed_g = recon_genes_s.cpu().detach().numpy()
        g_std_c = np.std(exp_mu, axis=1)
        g_std_trav_c = np.std(traversed_g, axis=1)
        diff_c = np.abs(traversed_g[:, -1, :] - traversed_g[:, 0, :])

        for arm in range(n_arm):
            V_g_pc = diff_c[arm, :] / g_std
            g_var_mean[i_cat, arm, 0, :] = V_g_pc

            plt.figure()
            plt.scatter(mu[arm, :, 0], mu[arm, :, 1])
            plt.scatter(smp[arm, :, 0], smp[arm, :, 1])
            plt.arrow(smp[arm, 0, 0], smp[arm, 0, 1],
                      smp[arm, -1, 0] - smp[arm, 0, 0], smp[arm, -1, 1] - smp[arm, 0, 1])
            plt.tight_layout()
            # # plt.savefig(saving_folder + f'/State/state_{cat}_arm_{arm}.png', dpi=600)
            #
            # fig, ax = plt.subplots(figsize=[15, 3])
            # n_points = len(g_idx) + (len(g_subset) - 1) * shift
            # x_range = np.arange(n_points)
            # # ax.plot(x_range, np.zeros(n_points), linestyle='--', linewidth=0.8, color='gray')
            # xticks = []
            # xtick_label = []
            # for ig, g_s in enumerate(g_subset):
            #     if ig > 0:
            #         x_s = np.arange(len(g_s)) + shift + x_s[-1]
            #     else:
            #         x_s = np.arange(len(g_s))
            #
            #     ax.errorbar(x_s, V_g_pc[g_s], yerr=0, linestyle='None', fmt='o', markersize=6, capsize=4, color=color[i_cat])
            #     # ax.fill_between(x_s, g_var_mean[g_s] - g_var_std[g_s], g_var_mean[g_s] + g_var_std[g_s], color=color[i_cat])
            #     xticks.append(x_s)
            #     g_label = data['gene_id'][g_s]
            #     zg_indx = np.where(sum_exp_c[g_s] == 0)[0]
            #     new_g = ['▶ ' + sg for sg in g_label[zg_indx]]
            #     g_label[zg_indx] = new_g
            #     xtick_label.append(g_label)
            #
            # # ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=True)
            # # ax.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True)
            # xticks = np.concatenate(xticks)
            # ax.set_xticks(xticks)
            # ax.set_xticklabels(np.concatenate(xtick_label), rotation=90, fontsize=12)
            # ax.set_xlabel('')
            # for tick in ax.get_yticklabels():
            #     tick.set_rotation(0)
            #     tick.set_fontsize(12)
            #
            # ax.set_ylabel(r'$V_{s}(g)$', fontsize=20)
            # # ax2 = ax.twinx()
            # # ax2.set_ylabel(c_ano[i_cat], fontsize=20, rotation=-90, labelpad=28)
            # # ax2.set_yticks([])
            # ax.set_xlim([xticks[0] - .5, xticks[-1] + .5])
            # ax.set_ylim([0, 2.])
            # ax.set_title(c_ano[i_cat], fontsize=20, pad=10)
            # plt.tight_layout()
            # # plt.savefig(saving_folder + f'/State/traversal_{cat}_pc_1_arm_{arm}.png', dpi=600)
            # plt.close('all')

    if mode == 'state':
        for s_i in range(state_dim):
            exp_mu = torch.stack(recon_m[s_i]).cpu().detach().numpy()
            traversed_g = torch.stack(recon_genes_s[s_i]).cpu().detach().numpy()
            g_std_c = np.std(exp_mu, axis=0)
            g_std_trav_c = np.std(traversed_g, axis=2)
            diff_c = np.abs(traversed_g[:, :, -1, :] - traversed_g[:, :, 0, :])

            mu_g_std_trav_c = np.mean(g_std_trav_c, axis=0)
            std_g_std_trav_c = np.std(g_std_trav_c, axis=0)
            mu_diff_c = np.mean(diff_c, axis=0)
            std_diff_c = np.std(diff_c, axis=0)
            # plt.figure()
            # for arm in range(n_arm):
            #     y_1 = mu_g_std_trav_c[arm, :] / g_std_c[arm, :]
            #     y_2 = mu_g_std_trav_c[arm, :] / g_std
            #     plt.scatter(x, y_1[g_idx], label=f'per c - arm {arm}')
            #     plt.scatter(x, y_2[g_idx], label=f'all - arm {arm}')
            #
            # plt.legend()

            for arm in range(n_arm):
                fig, ax = plt.subplots(figsize=[15, 3])
                n_points = len(g_idx) + (len(g_subset) - 1) * shift
                x_range = np.arange(n_points)
                ax.plot(x_range, np.zeros(n_points), linestyle='--', linewidth=0.8, color='gray')
                g_std_c_expand = np.outer(np.ones(diff_c[:, arm, :].shape[0]), g_std_c[arm, :])
                g_std_expand = np.outer(np.ones(diff_c[:, arm, :].shape[0]), g_std)
                g_var_c = diff_c[:, arm, :] / g_std_c_expand
                g_var = diff_c[:, arm, :] / g_std_expand
                V_s_g = np.mean(g_var, axis=0)
                dV_s_g = np.var(g_var, axis=0)
                g_var_mean[i_cat, arm, s_i, :] = V_s_g
                g_var_std[i_cat, arm, s_i, :] = dV_s_g

                # xticks = []
                # xtick_label = []
                # for ig, g_s in enumerate(g_subset):
                #     if ig > 0:
                #         x_s = np.arange(len(g_s)) + shift + x_s[-1]
                #     else:
                #         x_s = np.arange(len(g_s))
                #
                #     ax.errorbar(x_s, V_s_g[g_s], yerr=dV_s_g[g_s], linestyle='None', fmt='o', markersize=6, capsize=4, color=color[i_cat])
                #     # ax.fill_between(x_s, g_var_mean[g_s] - g_var_std[g_s], g_var_mean[g_s] + g_var_std[g_s], color=color[i_cat])
                #     xticks.append(x_s)
                #     g_label = data['gene_id'][g_s]
                #     zg_indx = np.where(sum_exp_c[g_s] == 0)[0]
                #     new_g = ['▶ ' + sg for sg in g_label[zg_indx]]
                #     g_label[zg_indx] = new_g
                #     xtick_label.append(g_label)
                #
                # # ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=True)
                # # ax.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True)
                # xticks = np.concatenate(xticks)
                # ax.set_xticks(xticks)
                # ax.set_xticklabels(np.concatenate(xtick_label), rotation=90, fontsize=12, style='italic')
                # ax.set_xlabel('')
                # for tick in ax.get_yticklabels():
                #     tick.set_rotation(0)
                #     tick.set_fontsize(12)
                #
                # if s_i == 0:
                #     ax.set_ylabel(r'$V_{s}(g)$', fontsize=20)
                # elif s_i == 1:
                #     ax.set_ylabel(r'$V_{s}(g)$', fontsize=20)
                #
                # # ax.yaxis.set_label_position("right")
                # ax2 = ax.twinx()
                # ax2.set_ylabel(c_ano[i_cat], fontsize=20, rotation=-90, labelpad=28)
                # ax2.set_yticks([])
                # ax.set_xlim([xticks[0] - .5, xticks[-1] + .5])
                # y_lim = np.max(np.abs(g_var_mean))
                # ax.set_ylim([-1*y_lim - .02, y_lim + .02])
                # ax.set_ylim([0, .45])
                # plt.tight_layout()
                # plt.savefig(saving_folder + f'/State/traversal_{cat}_ds_{s_i}_arm_{arm}.png', dpi=600)

distance_met = np.zeros((model_order, model_order))
for i_cat in range(model_order):
    for j_cat in range(i_cat+1, model_order):
        distance_met[i_cat, j_cat] = distance.euclidean(med_c_gene[i_cat, :], med_c_gene[j_cat, :])


sum_dict = dict()
sum_dict['V_g_mean'] = g_var_mean
sum_dict['V_g_std'] = g_var_std
sum_dict['g_subset'] = g_subset
sum_dict['c_cat'] = c_cat
sum_dict['pathways'] = signaling_pathways
sum_dict['s_mu'] = s_mu
sum_dict['s_travers'] = s_travers
f = open(saving_folder + f'/State/traversal_{mode}_K_{model_order}_2.pickle', "wb")
# f = open(saving_folder + f'/State/state_mu_{mode}_K_{model_order}.pickle', "wb")
pickle.dump(sum_dict, f)
f.close()

# for ig, g_s in enumerate(g_subset):
#     plt.figure()
#     plt.imshow(g_var_mean[:, 0, 1, g_s])
#     plt.title(signaling_pathways[ig])
#     plt.xticks(np.arange(len(g_s)), data['gene_id'][g_s], rotation=90)
#     plt.yticks(np.arange(model_order), c_cat)
#     plt.colorbar()
#     plt.savefig(saving_folder + f'/State/traversal_{cat}_ds_{s_i}_arm_{arm}.png', dpi=600)

