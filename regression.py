import os, glob, pickle
import sys
import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from numpy.linalg import norm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
import colorsys
from sklearn.metrics import confusion_matrix
from utils.training import train_cplmixVAE
from utils.eval_models import eval_mixmodel
from utils.config import load_config
from utils.get_region_info import get_roi
from utils.state_analysis import state_analyzer


metadata = 'ccf'
ccf_dim = 3

n_run = 2
temp = 1
subclass = 'L4-5_IT'
n_categories = 15
state_dim = 3
n_arm = 2
tau = .05
fc_dim = 100
latent_dim = 10
lr = 0.001
p_drop = .5
batch_size = 1000
n_epoch = 2000
n_epoch_p = 1000
n_gene = 10000
augmentation = True
device = None
ref_prior = False
sort_gene = True

paths = load_config(config_file='config.toml')
saving_folder = paths['package_dir'] / paths['saving_folder_cplmix']
data_file = paths['local_data_path'] / paths['data_' + subclass]

folder_name = f'{subclass}_run_{n_run}_K_{n_categories}_Sdim_{state_dim}_aug_{augmentation}_nGene_{n_gene}_' + \
              f'p_drop_{p_drop}_fc_dim_{fc_dim}_temp_1_lr_{lr}_n_arm_{n_arm}_tau_{tau}_lam_1_' + \
              f'nbatch_{batch_size}_' + f'nepoch_{n_epoch}_nepochP_{n_epoch_p}'

folder = str(saving_folder / folder_name)

selected_model = glob.glob(saving_folder + '/model/cpl_mixVAE_model_*')[0]

# selected_model = glob.glob(folder + '/model/cpl_mixVAE_model_after_pruning_' + str(n_categories - model_order) + '*')[0]

cpl_mixVAE, data, outcome = eval_mixmodel(data=[],
                                           n_categories=n_categories,
                                           state_dim=state_dim,
                                           n_arm=n_arm,
                                           latent_dim=latent_dim,
                                           fc_dim=fc_dim,
                                           n_gene=n_gene,
                                           tau=tau,
                                           saving_folder=saving_folder,
                                           ref_pc=ref_prior,
                                           subclass=subclass,
                                           device=device,
                                           sort_gene=sort_gene,
                                           loss_mode='ZINB',
                                           models=[selected_model])

visp_idx = np.where(data['region_label'] == 'VISp')[0]
vis_idx = np.where(data['region_label'] == 'VIS')[0]
data['region_label'][visp_idx] = data['region_label'][vis_idx[0]]
data['region_id'][visp_idx] = data['region_id'][vis_idx[0]]
data['region_color'][visp_idx] = data['region_color'][vis_idx[0]]

ccf = pd.read_csv(paths['package_dir'] / paths['ccf_file'])
x_c, y_c, z_c = (1320, 800, 1140)
data['ccf'] = np.zeros((len(data['region_label']), ccf_dim))
data['roi_weight'] = np.zeros(len(data['region_label']))
data['flat_ccf'] = np.zeros((len(data['region_label']), 2))
n_roi = len(np.unique(data['region_label']))
for roi in np.unique(data['region_label']):
    ind = np.where(data['region_label'] == roi)[0]
    x = ccf[ccf.roi == roi][ccf.layer == 'All'].x_l.values
    y = ccf[ccf.roi == roi][ccf.layer == 'All'].y_l.values
    z = ccf[ccf.roi == roi][ccf.layer == 'All'].z_l.values
    x_f = ccf[ccf.roi == roi][ccf.layer == 'All'].f_x_l.values
    y_f = ccf[ccf.roi == roi][ccf.layer == 'All'].f_y_l.values
    data['ccf'][ind, :] = np.vstack((x, y, z))[:, 0]
    data['flat_ccf'][ind, :] = np.vstack((x_f, y_f))[:, 0]
    data['roi_weight'][ind] = (1. - len(ind)/len(data['region_label'])) / (n_roi - 1)

ccf_norm = data['ccf'] - np.min(data['ccf'], axis=0)
ccf_norm = ccf_norm / np.max(ccf_norm, axis=0)

ccff_norm = data['flat_ccf'] - np.min(data['flat_ccf'], axis=0)
ccff_norm = ccff_norm / np.max(ccff_norm, axis=0)

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.naive_bayes import CategoricalNB, GaussianNB, BernoulliNB
from utils.data_tools import data_split

n_rn = 1000
arm = 0
kfold = 10
seed = 100
min_c = kfold * 10

df_roi = pd.DataFrame()
df_roi['region_label'] = data['region_label']
df_roi['region_id'] = data['region_id']
df_roi['region_color'] = data['region_color']

region_label, region_id, region_color = get_roi(df_roi)

print(np.unique(region_label), np.unique(region_id))

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(region_label)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
roi_onehot = onehot_encoder.fit_transform(integer_encoded)

mu = outcome['state_mu'][arm, :, :]
var = np.exp(outcome['state_var'][arm, :, :])
pred_type = outcome['pred_label'][-1][arm, :]
mismatch = outcome['pred_label'][-1][0, :] != outcome['pred_label'][-1][1, :]
cp_mean = np.mean((outcome['c_prob'][0, mismatch, :], outcome['c_prob'][1, mismatch, :]), axis=0)
pred_type[mismatch] = np.argmax(cp_mean, axis=1) + 1
pred_type = pred_type.astype(int)

color_list = np.unique(data['cluster_color'])
cat_color = np.copy(data['cluster_color'])
for ic, cc in enumerate(np.unique(pred_type)):
    c_idx = np.where(pred_type == cc)[0]
    cat_color[c_idx] = color_list[ic]

s_idx = outcome['sample_id'][-1].astype(int)
y = data['ccf'][s_idx]
region_weight = data['roi_weight'][s_idx]
y_norm = ccf_norm[s_idx]
yf = data['flat_ccf'][s_idx]
yf_norm = ccff_norm[s_idx]
color = region_color[s_idx]
tcolor = data['cluster_color'][s_idx]
t_type = data['cluster_label'][s_idx]
roi = region_label[s_idx]
xyz = data[metadata][s_idx]
subclass_label = data['subclass_label'][s_idx]
subclass_clr = data['subclass_color'][s_idx]
cat_list = np.unique(pred_type)

acc_rf_c, balanc_acc_rf_c = np.zeros(n_categories), np.zeros(n_categories)
acc_mlp_c, balanc_acc_mlp_c = np.zeros(n_categories), np.zeros(n_categories)
acc_nb_c, balanc_acc_nb_c = np.zeros(n_categories), np.zeros(n_categories)
chance_level = np.zeros(n_categories)
conf_mat_mlp = [[] for nc in range(n_categories)]
conf_mat_nb = [[] for nc in range(n_categories)]

for sc in cat_list:
    print(sc)
    saving_folder = paths['package_dir'] / paths['saving_folder_cplmix']
    os.makedirs(saving_folder / paths['cplmix_glum'] / str(sc), exist_ok=True)
    saving_folder = str(saving_folder / paths['cplmix_glum'] / str(sc))

    sc_idx = np.where(pred_type == sc)[0]

    y_sc = np.ravel(integer_encoded[sc_idx])
    mu_c = mu[sc_idx, :]
    var_c = var[sc_idx, :]
    type_sc = t_type[sc_idx]
    pred_type_sc = pred_type[sc_idx]
    color_sc = color[sc_idx]
    tcolor_sc = tcolor[sc_idx]
    ccolor_sc = cat_color[sc_idx]
    roi_sc = roi[sc_idx]
    xyz_sc = xyz[sc_idx]

    kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)
    acc = dict()
    pred_labels = dict()
    ref_labels = dict()

    uniq_c = np.unique(y_sc)
    min_p = 0.8 / (len(uniq_c))
    num_c = np.array([sum(y_sc == ll) for ll in uniq_c])
    class_p = num_c / len(y_sc)

    new_x = []
    new_label = []
    for i_cl, cl in enumerate(uniq_c):
        if num_c[i_cl] < min_c or class_p[i_cl] < min_p:
            n_smp = int(np.max((min_c, .5 * np.max(num_c))))
            new_x_c = np.zeros((n_smp, mu_c.shape[1]))
            mean_mu = np.mean(mu_c[y_sc == cl, :], axis=0)
            mean_var = np.mean(var_c[y_sc == cl, :], axis=0)
            for sd in range(mu_c.shape[1]):
                new_x_c[:, sd] = np.random.normal(mean_mu[sd], 0.1 * mean_var[sd], size=n_smp)

            new_x.append(new_x_c)
            new_label.append(cl * np.ones(n_smp))

    if len(new_x) > 0:
        new_x = np.concatenate(new_x)
        new_label = np.concatenate(new_label)
        x_sc = np.concatenate((mu_c, new_x), axis=0)
        y_sc = np.concatenate((y_sc, new_label), axis=0)


    else:
        x_sc = mu[sc_idx, :]

    num_c = np.array([sum(y_sc == ll) for ll in uniq_c])
    class_p = num_c / len(y_sc)
    print(class_p)

    acc_rf, blnc_acc_rf = [], []
    acc_mlp, blnc_acc_mlp = [], []
    acc_nb, blnc_acc_nb = [], []
    pred_mlp_roi, pred_nb_roi, true_roi = [], [], []

    tr_ind, ts_ind = data_split(x=x_sc, nfold=kfold, label=y_sc)

    for train_index, test_index in zip(tr_ind, ts_ind):
        rfc = RandomForestClassifier()
        rfc.fit(x_sc[train_index, :], y_sc[train_index])
        y_pred = rfc.predict(x_sc[test_index, :])
        acc_rf.append(accuracy_score(y_sc[test_index], y_pred))
        blnc_acc_rf.append(balanced_accuracy_score(y_sc[test_index], y_pred))

        mlp = MLPClassifier(max_iter=2000)
        mlp.fit(x_sc[train_index, :], y_sc[train_index])
        y_pred = mlp.predict(x_sc[test_index, :])
        acc_mlp.append(accuracy_score(y_sc[test_index], y_pred))
        blnc_acc_mlp.append(balanced_accuracy_score(y_sc[test_index], y_pred))
        pred_mlp_roi.append(y_pred)

        nb = BernoulliNB(class_prior=class_p)
        nb.fit(x_sc[train_index, :], y_sc[train_index])
        y_pred = nb.predict(x_sc[test_index, :])
        acc_nb.append(accuracy_score(y_sc[test_index], y_pred))
        blnc_acc_nb.append(balanced_accuracy_score(y_sc[test_index], y_pred))
        pred_nb_roi.append(y_pred)
        true_roi.append(y_sc[test_index])

    pred_mlp_roi = np.concatenate(pred_mlp_roi)
    pred_nb_roi = np.concatenate(pred_nb_roi)
    true_roi = np.concatenate(true_roi)
    conf_mat_mlp[sc - 1].append(confusion_matrix(true_roi, pred_mlp_roi, normalize='true'))
    conf_mat_nb[sc - 1].append(confusion_matrix(true_roi, pred_nb_roi, normalize='true'))

    acc_rf_c[sc - 1] = np.mean(acc_rf)
    balanc_acc_rf_c[sc - 1] = np.mean(blnc_acc_rf)
    acc_mlp_c[sc - 1] = np.mean(acc_mlp)
    balanc_acc_mlp_c[sc - 1] = np.mean(blnc_acc_mlp)
    acc_nb_c[sc - 1] = np.mean(acc_nb)
    balanc_acc_nb_c[sc - 1] = np.mean(blnc_acc_nb)
    chance_level[sc - 1] = np.max(class_p)

    uniq_id = np.unique(np.ravel(integer_encoded[sc_idx]))
    order_roi = np.array([roi_sc[np.ravel(integer_encoded[sc_idx]) == i][0] for i in uniq_id])
    fig, axs = plt.subplots(1, 1, figsize=(15, 12), dpi=300)
    sns.set(font_scale=1.)
    sns.heatmap(conf_mat_mlp[sc - 1][0], xticklabels=order_roi, yticklabels=order_roi,
                vmin=0, vmax=1, ax=axs, cmap='Blues', cbar_kws={"shrink": 1})
    axs.set_ylabel('True ROI', fontsize=20)
    axs.set_xlabel('Pred. ROI', fontsize=20)
    axs.set_title(f'ACC: {balanc_acc_mlp_c[sc - 1]}')
    fig.tight_layout()
    fig.savefig(saving_folder + f'/confmtx_MOR_{model_order}_{sc}_roi_1.pdf', dpi=1000)


saving_folder = str(paths['package_dir'] / paths['saving_folder_cplmix'] / paths['cplmix_glum'])
width = 0.4
fig = plt.subplots(figsize =(15, 5))
p1 = plt.bar(np.arange(1, n_categories+1) - width/2, chance_level, width, label='Chance')
p2 = plt.bar(np.arange(1, n_categories+1) + width/2, balanc_acc_mlp_c, width, label='Prediction')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1, n_categories+1), np.arange(1, n_categories+1), rotation='vertical', fontsize=8)
plt.xlim([0.1, n_categories+0.5])
plt.ylim([0, 1])
plt.legend(loc='upper right')
fig.tight_layout()
fig.savefig(saving_folder + f'/confmtx_MOR_{model_order}_{sc}_roi_1.pdf', dpi=1000)

fig = plt.subplots(figsize =(15, 5))
p1 = plt.bar(np.arange(1, n_categories+1) - width/2, chance_level, width, label='Chance')
p2 = plt.bar(np.arange(1, n_categories+1) + width/2, acc_nb_c, width, label='Prediction')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1, n_categories+1), np.arange(1, n_categories+1), rotation='vertical', fontsize=8)
plt.xlim([0.1, n_categories+0.5])
plt.ylim([0, 1])
plt.legend(loc='upper right')

fig = plt.subplots(figsize =(15, 5))
p1 = plt.bar(np.arange(1, n_categories+1) - width/2, balanc_acc_mlp_c, width, label='mlp')
p2 = plt.bar(np.arange(1, n_categories+1) + width/2, balanc_acc_nb_c, width, label='nb')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1, n_categories+1), np.arange(1, n_categories+1), rotation='vertical', fontsize=8)
plt.xlim([0.1, n_categories+0.5])
plt.ylim([0, 1])
plt.legend(loc='upper right')