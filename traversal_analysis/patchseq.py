from cplmix.utils.training import train_cplmixVAE, train_crossModal
from cplmix.utils.helpers import load_patchseq
import torch
from cplmix.utils.helpers import load_config
from cplmix.utils.state_analysis import state_analyzer
from cplmix.run.eval_cplMixVAE import eval_model
from cplmix.utils.gene_analysis import traversal_analysis, get_gene_indx, genes_subset
import matplotlib.pyplot as plt
import toml
import glob, os
import numpy as np
import pandas as pd
import pickle
from scipy.spatial import distance

train = True
n_sample = 100
n_run = 0
n_categories = 100
state_dim = dict()
state_dim['T'] = 2
state_dim['E'] = 3
n_arm = dict()
n_arm['T'] = 1
n_arm['E'] = 1
temp = 1
device = None
using_mask = True
min_n_cell = 1000
shift = 3


selected_E_features = ['ap_1_downstroke_0_long_square', 'ap_1_downstroke_short_square',
       'ap_1_fast_trough_v_0_long_square',
       'ap_1_fast_trough_v_short_square', 'ap_1_peak_v_0_long_square',
       'ap_1_peak_v_short_square', 'ap_1_threshold_v_0_long_square',
       'ap_1_threshold_v_short_square', 'ap_1_upstroke_0_long_square',
       'ap_1_upstroke_downstroke_ratio_0_long_square',
       'ap_1_upstroke_downstroke_ratio_short_square',
       'ap_1_upstroke_short_square', 'ap_1_width_0_long_square',
       'ap_1_width_short_square', 'avg_rate_0_long_square',
       'input_resistance', 'latency_0_long_square', 'rheobase_i',
       'sag_measured_at', 'sag_nearest_minus_100', 'short_square_current',
       'stimulus_amplitude_0_long_square', 'tau', 'v_baseline', 'isi_shape_0']

E_label = ['AP1_DS_LS', 'AP1_DS_SS', 'AP1_Fast_LS', 'AP1_Fast_SS', 'AP1_Peak_LS',
       'AP1_Peak_SS', 'AP1_thr_LS', 'AP1_thr_SS', 'AP1_UP_LS', 'AP1_UP_SS',
       'AP1_Ratio_LS', 'AP1_ratio_SS', 'AP1_W_LS', 'AP1_W_SS', 'Avg_Rate_LS',
       'Input_R', 'Latency_LS', 'rheobase_i', 'sag_meas', 'sag_near', 'SS_i',
       'Amp_LS', 'tau', 'V_BL', 'ISI_shape']

paths = load_config(config_file='config.toml')
saving_folder = paths['package_dir'] / 'results'
data_folder = paths['package_dir'] / paths['data']
folder_name = f"run_{n_run}_K_{n_categories}_SdimT_{state_dim['T']}_SdimE_{state_dim['E']}_lr_0.001_n_armT_" + \
              f"{n_arm['T']}_n_armE_{n_arm['E']}_tau_0.01_nbatch_1000_nepoch_10000_nepochP_10000"

saving_folder = str(saving_folder / folder_name)

model_order = 60
selected_model = glob.glob(saving_folder + '/model/cpl_mixVAE_model_after_pruning_' + str(n_categories - model_order) + '*')[0]
# selected_model = glob.glob(saving_folder + '/model/cpl_mixVAE_model_*')[0]
cpl_mixVAE, data, outcome = eval_model(n_categories=n_categories,
                                       state_dim_T=state_dim['T'],
                                       state_dim_E=state_dim['E'],
                                       n_arm_T=n_arm['T'],
                                       n_arm_E=n_arm['E'],
                                       fc_dim_T=100,
                                       fc_dim_E=100,
                                       device=device,
                                       all_data=True,
                                       saving_folder=saving_folder,
                                       folder_name=folder_name,
                                       data_path=data_folder,
                                       using_mask=using_mask,
                                       models=[selected_model])

consensus = dict()
consensus['TE'] = outcome['consensus'][0]['TE'][0]
mask = dict()
mask['TE'] = data['crossModal_id'] #[np.where(T_smp_id == c_id)[0][0] for c_id in crossModal_id]
mod = ['T', 'E']

arm = 0
prob_T = np.expand_dims(outcome['c_prob']['T'][arm], axis=2)
prob_E = np.expand_dims(outcome['c_prob']['E'][arm], axis=2)
prob_TE = np.concatenate((prob_T, prob_E), axis=2)
prob_TE = np.squeeze(np.mean(prob_TE, axis=2))
c_label = np.argmax(prob_TE, axis=1) + 1
c_cat = np.unique(c_label)
pruning_mask = outcome['nprune_indx']

x = dict()
feature, n_feature = dict(), dict()
x['T'] = data['XT'][mask['TE']]
x['E'] = data['XE'][mask['TE']]
feature['E'] = data['E_features']
feature['T'] = data['gene_ids']
n_feature['T'] = len(data['gene_ids'])
n_feature['E'] = len(data['E_features'])
t_type = data['cluster_label']

ieg, hkg, NPP_genes, NPGPCR_genes = genes_subset()

g_subset = []
g_subset.append(get_gene_indx(ref_gene=data['gene_ids'], traget_gene=NPP_genes))
g_subset.append(get_gene_indx(ref_gene=data['gene_ids'], traget_gene=NPGPCR_genes))
# g_subset.append(get_gene_indx(ref_gene=data['gene_id'], traget_gene=ieg))
# g_subset.append(get_gene_indx(ref_gene=data['gene_id'], traget_gene=hkg))

kegg_pathway_genes = toml.load(paths['package_dir'] / 'KEGG.toml')
kegg_pathway = pd.read_csv(paths['package_dir'] / 'KEGG_pathways.csv')

g_id_set = []
g_id_set.append(NPP_genes)
g_id_set.append(NPGPCR_genes)
for pathway in kegg_pathway.pathway_id:
    gg = kegg_pathway_genes[pathway]['ps_genes']
    g_id_set.append(gg)
    g_subset.append(get_gene_indx(ref_gene=data['gene_ids'], traget_gene=gg))

signaling_pathways = np.concatenate((['NPP', 'NP-GPCR'], kegg_pathway.pathway_name.values))
g_idx = np.concatenate(g_subset).astype(int)
# g_subset = [NPPg_idx, NPGPCRg_idx, ieg_idx, hkg_idx]
selected_genes = data['gene_ids'][g_idx]

f_std = dict()
f_std['T'] = np.std(x['T'], axis=0)
f_std['E'] = np.std(x['E'], axis=0)

chance_level = 0.0
corr_mean = dict()
corr_std = dict()
p_val_mean = dict()
corr = dict()
n_cell = dict()
ref_type = dict()
delta_g = dict()
d_feature = dict()
mod = ['T', 'E']

c_ano = ['Lamp5 TE_17', 'Lamp5 TE_16', 'Pvalb TE_60', 'Sst TE_41', 'Sst TE_39', 'L6 IT TE_6', 'L6 IT TE_7', 'L4 IT TE_2']
selected_c = [76, 50, 57, 49, 60, 72, 75, 91]
color = ['#FF7290', '#DD6091', '#AF3F64', '#D6C300', '#CC6D3D', '#A19922', '#A19922', '#00979D']

med_c_gene = dict()
distance_met = dict()
g_var_mean = dict()
g_var_std = dict()

scale = dict()
scale['T'] = 10
scale['E'] = 0.5
state_mode = 'pca'

for m in mod:
    if m == 'E':
        ef_indx = [np.where(data['E_features'] == sf)[0][0] for sf in selected_E_features]

    delta_g[m] = np.zeros((model_order, state_dim[m], len(feature[m]), n_arm[m]))
    corr_mean[m] = np.zeros((model_order, state_dim[m], len(feature[m]), n_arm[m]))
    corr_std[m] = np.zeros((model_order, state_dim[m], len(feature[m]), n_arm[m]))
    p_val_mean[m] = np.zeros((model_order, state_dim[m], len(feature[m]), n_arm[m]))
    corr[m] = np.zeros((model_order, state_dim[m], min_n_cell, len(feature[m]), n_arm[m]))
    d_feature[m] = np.zeros((model_order, state_dim[m], min_n_cell, len(feature[m]), n_arm[m]))
    g_var_mean[m] = np.zeros((model_order, n_arm[m], state_dim[m], len(feature[m])))
    g_var_std[m] = np.zeros((model_order, n_arm[m], state_dim[m], len(feature[m])))
    n_cell[m] = np.zeros(model_order)
    ref_type[m] = []
    med_c_gene[m] = np.zeros((model_order, n_feature[m]))
    distance_met[m] = np.zeros((model_order, model_order))
    s_mu, s_travers = [], []

    for i_cat, cat in enumerate(c_cat):

        indx = np.where(c_label==cat)[0]
        print(m, cat, len(indx))
        ref_type[m].append(np.unique(t_type[indx]))
        med_c_gene[m][i_cat, :] = np.mean(x[m][indx, :], axis=0)

        recon_genes_s, recon_m, state_s, state_mu = traversal_analysis(data=x[m][indx, :], model=cpl_mixVAE, state_dim=state_dim[m],
                            mod=m, mask=pruning_mask, temp=temp, min_n_cell=min_n_cell, scale=scale[m], mode=state_mode)

        ref_t = np.unique(t_type[indx])
        num_ttype = [sum(t_type[indx] == rf) for rf in ref_t]
        ref_t = ref_t[np.argsort(num_ttype)][::-1]
        n_cell[m][i_cat] = len(recon_genes_s[0])
        n_c = len(recon_genes_s[0])

        sum_feature_c = np.sum(x[m][indx, :], axis=0)

        if state_mode == 'pca':
            mu = state_mu.cpu().detach().numpy()
            smp = state_s.cpu().detach().numpy()
            s_mu.append(mu)
            s_travers.append(smp)
            exp_mu = recon_m.cpu().detach().numpy()
            traversed_g = recon_genes_s.cpu().detach().numpy()
            g_std_c = np.std(exp_mu, axis=1)
            g_std_trav_c = np.std(traversed_g, axis=1)
            diff_c = np.abs(traversed_g[:, -1, :] - traversed_g[:, 0, :])

            for arm in range(n_arm[m]):
                V_g_pc = diff_c[arm, :] / f_std[m]
                g_var_mean[m][i_cat, arm, 0, :] = V_g_pc

                # if m == 'T':
                #     # plt.figure()
                #     # plt.scatter(mu[arm, :, 0], mu[arm, :, 1])
                #     # plt.scatter(smp[arm, :, 0], smp[arm, :, 1])
                #     # plt.arrow(smp[arm, 0, 0], smp[arm, 0, 1], smp[arm, -1, 0] - smp[arm, 0, 0], smp[arm, -1, 1] - smp[arm, 0, 1])
                #     # plt.tight_layout()
                #     # plt.savefig(saving_folder + f'/State/state_{cat}_arm_{arm}.png', dpi=600)
                #
                #     fig, ax = plt.subplots(figsize=[15, 3])
                #     n_points = len(g_idx) + (len(g_subset) - 1) * shift
                #     x_range = np.arange(n_points)
                #     # ax.plot(x_range, np.zeros(n_points), linestyle='--', linewidth=0.8, color='gray')
                #     xticks = []
                #     xtick_label = []
                #     for ig, g_s in enumerate(g_subset):
                #         if ig > 0:
                #             x_s = np.arange(len(g_s)) + shift + x_s[-1]
                #         else:
                #             x_s = np.arange(len(g_s))
                #
                #         ax.errorbar(x_s, V_g_pc[g_s], yerr=0, linestyle='None', fmt='o', markersize=6, capsize=4, color=color[i_cat])
                #         # ax.fill_between(x_s, g_var_mean[g_s] - g_var_std[g_s], g_var_mean[g_s] + g_var_std[g_s], color=color[i_cat])
                #         xticks.append(x_s)
                #         g_label = data['gene_ids'][g_s]
                #         zg_indx = np.where(sum_feature_c[g_s] == 0)[0]
                #         new_g = ['▶ ' + sg for sg in g_label[zg_indx]]
                #         g_label[zg_indx] = new_g
                #         xtick_label.append(g_label)
                #
                #     xticks = np.concatenate(xticks)
                #     ax.set_xticks(xticks)
                #     ax.set_xticklabels(np.concatenate(xtick_label), rotation=90, fontsize=12)
                #     ax.set_xlabel('')
                #     for tick in ax.get_yticklabels():
                #         tick.set_rotation(0)
                #         tick.set_fontsize(12)
                #
                #     ax.set_ylabel(r'$V_{s}(g)$', fontsize=20)
                #     # ax2 = ax.twinx()
                #     # ax2.set_ylabel(c_ano[i_cat], fontsize=20, rotation=-90, labelpad=28)
                #     # ax2.set_yticks([])
                #     ax.set_xlim([xticks[0] - .5, xticks[-1] + .5])
                #     ax.set_ylim([0, 2.])
                #     ax.set_title(c_ano[i_cat], fontsize=20, pad=10)
                #     plt.tight_layout()
                #     plt.savefig(saving_folder + f'/State_{m}/traversal_{cat}_pc_1.png', dpi=600)
                #     plt.close('all')

                # if m == 'E':
                #     fig, ax = plt.subplots(figsize=[15, 4])
                #     x_range = np.arange(len(ef_indx))
                #     # ax.plot(x_range, np.zeros(len(ef_indx)), linestyle='--', linewidth=0.8, color='gray')
                #     ax.errorbar(x_range, V_g_pc[ef_indx], yerr=0, linestyle='None', fmt='o', markersize=6, capsize=4, color=color[i_cat])
                #     xticks = []
                #     xtick_label = []
                #
                #     xticks.append(x_range)
                #     xtick_label.append(E_label)
                #
                #     # ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=True)
                #     # ax.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True)
                #     xticks = np.concatenate(xticks)
                #     ax.set_xticks(xticks)
                #     ax.set_xticklabels(np.concatenate(xtick_label), rotation=90, fontsize=16)
                #     ax.set_xlabel('')
                #     for tick in ax.get_yticklabels():
                #         tick.set_rotation(0)
                #         tick.set_fontsize(14)
                #
                #     ax.set_ylabel(r'$V_{s}(e)$', fontsize=24, labelpad=20)
                #
                #     # ax.yaxis.set_label_position("right")
                #     ax2 = ax.twinx()
                #     ax2.set_ylabel(c_ano[i_cat], fontsize=24, rotation=-90, labelpad=32)
                #     ax2.set_yticks([])
                #     ax.set_xlim([xticks[0] - .5, xticks[-1] + .5])
                #     # y_lim = np.max(np.abs(V_g_pc))
                #     # ax.set_ylim([-1 * y_lim - .02, y_lim + .02])
                #     ax.set_ylim([0, 1])
                #     plt.tight_layout()
                #     plt.savefig(saving_folder + f'/State_{m}/traversal_{cat}_pc_arm_{arm}.png', dpi=600, bbox_inches='tight')

        if state_mode == 'state':
            for s_i in range(state_dim[m]):

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

                for arm in range(n_arm[m]):

                    g_std_c_expand = np.outer(np.ones(diff_c[:, arm, :].shape[0]), g_std_c[arm, :])
                    g_std_expand = np.outer(np.ones(diff_c[:, arm, :].shape[0]), f_std[m])
                    g_var_c = diff_c[:, arm, :] / g_std_c_expand
                    g_var = diff_c[:, arm, :] / g_std_expand
                    g_var_mean = np.mean(g_var, axis=0)
                    g_var_std = np.var(g_var, axis=0)


                    if m == 'T':
                        fig, ax = plt.subplots(figsize=[15, 3])
                        n_points = len(g_idx) + (len(g_subset) - 1) * shift
                        x_range = np.arange(n_points)
                        # ax.plot(x_range, np.zeros(n_points), linestyle='--', linewidth=0.8, color='gray')
                        xticks = []
                        xtick_label = []
                        for ig, g_s in enumerate(g_subset):
                            if ig > 0:
                                x_s = np.arange(len(g_s)) + shift + x_s[-1]
                            else:
                                x_s = np.arange(len(g_s))

                            ax.errorbar(x_s, g_var_mean[g_s], yerr=g_var_std[g_s], linestyle='None', fmt='o', markersize=6,
                                        capsize=4, color=color[i_cat])
                            xticks.append(x_s)
                            g_label = feature[m][g_s]
                            zg_indx = np.where(sum_feature_c[g_s] == 0)[0]
                            new_g = ['▶ ' + sg for sg in g_label[zg_indx]]
                            g_label[zg_indx] = new_g
                            xtick_label.append(g_label)

                        xticks = np.concatenate(xticks)
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(np.concatenate(xtick_label), rotation=90, fontsize=12)
                        ax.set_xlabel('')
                        for tick in ax.get_yticklabels():
                            tick.set_rotation(0)
                            tick.set_fontsize(12)

                        ax.set_ylabel(r'$V_{s}(g)$', fontsize=20, labelpad=20)

                            # ax.yaxis.set_label_position("right")
                        ax2 = ax.twinx()
                        ax2.set_ylabel(c_ano[i_cat], fontsize=20, rotation=-90, labelpad=32)
                        ax2.set_yticks([])
                        ax.set_xlim([xticks[0] - .5, xticks[-1] + .5])
                        y_lim = np.max(np.abs(g_var_mean))
                        # ax.set_ylim([-1 * y_lim - .02, y_lim + .02])
                        ax.set_ylim([0, .5])
                        plt.tight_layout()
                        plt.savefig(saving_folder + f'/State_{m}/traversal_{cat}_ds_{s_i}_arm_{arm}.png', dpi=600, bbox_inches='tight')


                    if m == 'E':
                        fig, ax = plt.subplots(figsize=[15, 4])
                        x_range = np.arange(len(ef_indx))
                        # ax.plot(x_range, np.zeros(len(ef_indx)), linestyle='--', linewidth=0.8, color='gray')
                        ax.errorbar(x_range, g_var_mean[ef_indx], yerr=g_var_std[ef_indx], linestyle='None', fmt='o',
                                    markersize=6, capsize=4, color=color[i_cat])
                        xticks = []
                        xtick_label = []

                        xticks.append(x_range)
                        xtick_label.append(E_label)

                        # ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=True)
                        # ax.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True)
                        xticks = np.concatenate(xticks)
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(np.concatenate(xtick_label), rotation=90, fontsize=16)
                        ax.set_xlabel('')
                        for tick in ax.get_yticklabels():
                            tick.set_rotation(0)
                            tick.set_fontsize(14)


                        ax.set_ylabel(r'$V_{s}(e)$', fontsize=24, labelpad=20)

                        # ax.yaxis.set_label_position("right")
                        ax2 = ax.twinx()
                        ax2.set_ylabel(c_ano[i_cat], fontsize=24, rotation=-90, labelpad=32)
                        ax2.set_yticks([])
                        ax.set_xlim([xticks[0] - .5, xticks[-1] + .5])
                        y_lim = np.max(np.abs(g_var_mean))
                        # ax.set_ylim([-1 * y_lim - .02, y_lim + .02])
                        ax.set_ylim([0, 1])
                        plt.tight_layout()
                        plt.savefig(saving_folder + f'/State_{m}/traversal_{cat}_ds_{s_i}_arm_{arm}.png', dpi=600, bbox_inches='tight')

    # for i_cat in range(model_order):
    #     for j_cat in range(i_cat+1, model_order):
    #         distance_met[m][i_cat, j_cat] = distance.euclidean(med_c_gene[m][i_cat, :], med_c_gene[m][j_cat, :])
    #         distance_met[m][j_cat, i_cat] = distance.euclidean(med_c_gene[m][i_cat, :], med_c_gene[m][j_cat, :])
    #
    #


sum_dict = dict()
sum_dict['V_g_mean'] = g_var_mean
sum_dict['V_g_std'] = g_var_std
sum_dict['g_subset'] = g_subset
sum_dict['c_cat'] = c_cat
sum_dict['pathways'] = signaling_pathways
f = open(saving_folder + f'/traversal_{state_mode}_K_{model_order}_2.pickle', "wb")
# f = open(saving_folder + f'/State/state_mu_{mode}_K_{model_order}.pickle', "wb")
pickle.dump(sum_dict, f)
f.close()


