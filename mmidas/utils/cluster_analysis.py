import scipy.io as sio
import numpy as np
import os, pickle, glob
import seaborn as sns
from sklearn import decomposition
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def RF_classifier(data, labels, kfold, seed):

    kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)
    acc = dict()
    pred_labels = dict()
    ref_labels = dict()

    for ik, key in enumerate(labels.keys()):
        y = labels[key]
        acc[key] = []
        pred_labels[key] = []
        ref_labels[key] = []

        for train_index, test_index in kf.split(data):
            rfc = RandomForestClassifier()
            rfc.fit(data[train_index, :], y[train_index])
            y_pred = rfc.predict(data[test_index, :])
            acc[key].append(accuracy_score(y[test_index], y_pred))
            pred_labels[key].append(y_pred)
            ref_labels[key].append(y[test_index])

    return acc, ref_labels, pred_labels


def LDA_classifier(data, labels, kfold, seed):

    kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)
    acc = dict()
    pred_labels = dict()
    ref_labels = dict()

    for ik, key in enumerate(labels.keys()):
        y = labels[key]
        acc[key] = []
        pred_labels[key] = []
        ref_labels[key] = []

        for train_index, test_index in kf.split(data):
            lda = LinearDiscriminantAnalysis(store_covariance=True)
            lda.fit(data[train_index, :], y[train_index])
            y_pred = lda.predict(data[test_index, :])
            acc[key].append(accuracy_score(y[test_index], y_pred))
            pred_labels[key].append(y_pred)
            ref_labels[key].append(y[test_index])

    return acc, ref_labels, pred_labels


def QDA_classifier(data, labels, kfold, seed):

    kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)
    acc = dict()
    pred_labels = dict()
    ref_labels = dict()

    for ik, key in enumerate(labels.keys()):
        y = labels[key]
        acc[key] = []
        pred_labels[key] = []
        ref_labels[key] = []

        for train_index, test_index in kf.split(data):
            qda = QuadraticDiscriminantAnalysis(reg_param=1e-2, store_covariance=True)
            qda.fit(data[train_index, :], y[train_index])
            y_pred = qda.predict(data[test_index, :])
            acc[key].append(accuracy_score(y[test_index], y_pred))
            pred_labels[key].append(y_pred)
            ref_labels[key].append(y[test_index])

    return acc, ref_labels, pred_labels



def cluster_compare(data, labels, num_pc=0, saving_path=''):

    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot()

    if num_pc > 0:
        pca = PCA(n_components=num_pc)
        z = pca.fit(data).transform(data)
        silh_smp_score, sil_score = [], []
        c_size = []
        for ik, key in enumerate(labels.keys()):
            y = labels[key]
            uni_class = np.unique(y)
            sample_score = silhouette_samples(z, y)
            sil_score.append(silhouette_score(z, y))
            mean_smp_sc = np.zeros(len(uni_class))
            cluster_size = np.zeros(len(uni_class))
            for ic, c in enumerate(np.unique(y)):
                label_ind = np.where(labels[key] == c)[0]
                mean_smp_sc[ic] = np.mean(sample_score[label_ind])
                cluster_size[ic] = len(label_ind)

            silh_smp_score.append(mean_smp_sc)
            sort_indx = np.argsort(mean_smp_sc)
            c_size.append(cluster_size[sort_indx])
            ax.plot(np.arange(len(uni_class)), mean_smp_sc[sort_indx], label=key)

        ax.set_title(str(num_pc) + ' PCs', fontsize=18)
        ax.set_xlabel('Ordered clusters')
        ax.legend(prop={'size': 12})
        ax.set_ylabel('Ave. Silhouette scores')
        fig.tight_layout()

    return fig, silh_smp_score, sil_score, c_size


def K_selection(cplmixVAE_data, num_category, n_arm):

    n_comb = max(n_arm * (n_arm - 1) / 2, 1)

    with sns.axes_style("darkgrid"):
        # cplmixVAE_data = pickle.load(open(cplmixVAE_model, 'rb'))
        cplmixVAE_data['num_pruned'] = np.array(cplmixVAE_data['num_pruned'])
        cplmixVAE_data['dz'] = np.array(cplmixVAE_data['dz'])
        cplmixVAE_data['d_qz'] = np.array(cplmixVAE_data['d_qz'])
        cplmixVAE_data['con_min'] = np.array(cplmixVAE_data['con_min'])
        cplmixVAE_data['con_min'] = np.reshape(cplmixVAE_data['con_min'], (int(n_comb), len(cplmixVAE_data['d_qz'])))
        cplmixVAE_data['con_mean'] = np.array(cplmixVAE_data['con_mean'])
        cplmixVAE_data['con_mean'] = np.reshape(cplmixVAE_data['con_mean'], (int(n_comb), len(cplmixVAE_data['d_qz'])))
        indx = np.argsort(cplmixVAE_data['num_pruned'])
        recon_loss = []
        norm_recon = []

        for a in range(n_arm):
            recon_loss.append(np.array(cplmixVAE_data['recon_loss'][a]))
            # print(np.min(recon_loss[a]),  np.max(recon_loss[a]))
            tmp = recon_loss[a] - np.min(recon_loss[a])
            norm_recon.append(tmp / np.max(tmp))
            # norm_recon.append(recon_loss[a])

        norm_recon_mean = np.mean(norm_recon, axis=0)
        neg_cons = 1 - np.mean(cplmixVAE_data['con_mean'], axis=0)
        mean_cost = (neg_cons + norm_recon_mean) / 2 # cplmixVAE_data['d_qz']
        cost = []

        fig_1 = plt.figure(figsize=[10, 5])
        for a in range(n_arm):
            cost.append((cplmixVAE_data['d_qz'] + norm_recon[a] + neg_cons))
            ax1 = fig_1.add_subplot()
            ax1.plot(cplmixVAE_data['num_pruned'][indx], recon_loss[a][indx], label='Norm. Recon. Error')

        ax1.plot(cplmixVAE_data['num_pruned'][indx], cplmixVAE_data['d_qz'][indx], label='Dist. q(c)')
        ax1.plot(cplmixVAE_data['num_pruned'][indx], neg_cons[indx], label='Mean -Consensus')
        ax1.set_xlim([np.min(cplmixVAE_data['num_pruned'][indx])-1, num_category + 1])
        ax1.set_xlabel('Categories')
        ax1.set_xticks(cplmixVAE_data['num_pruned'][indx])
        ax1.set_xticklabels(cplmixVAE_data['num_pruned'][indx], fontsize=8, rotation=90)
        ax1.legend(loc='upper left')
        ax1.grid(b=True, which='major', linestyle='-')

        sdt = np.std(np.array(cost), axis=0)
        fig_2 = plt.figure(figsize=[10, 5])
        ax2 = fig_2.add_subplot()
        ax2.plot(cplmixVAE_data['num_pruned'][indx], mean_cost[indx], label=str(n_arm) + ' arms', c='black')
        ax2.fill_between(cplmixVAE_data['num_pruned'][indx], mean_cost[indx] - sdt[indx], mean_cost[indx] + sdt[indx], alpha=0.3, facecolor='black')
        ax2.set_xticks(cplmixVAE_data['num_pruned'][indx])
        ax2.set_xticklabels(cplmixVAE_data['num_pruned'][indx], fontsize=8, rotation=90)
        ax2.set_ylabel('Norm. Ave. Cost')
        ax2.set_xlabel('Categories')
        ax2.legend()
        ax2.grid()

        fig_3 = plt.figure(figsize=[10, 5])
        ax3 = fig_3.add_subplot()
        ax3.plot(cplmixVAE_data['num_pruned'][indx], np.mean(cplmixVAE_data['con_min'], axis=0)[indx], label='Min Consensus')
        ax3.plot(cplmixVAE_data['num_pruned'][indx], cplmixVAE_data['d_qz'][indx], label='Dist. q(c)')
        ax3.plot(cplmixVAE_data['num_pruned'][indx], neg_cons[indx], label='Mean Consensus')
        ax3.set_xlim([np.min(cplmixVAE_data['num_pruned'][indx]) - 1, num_category + 1])
        ax3.set_xlabel('Categories')
        ax3.set_xticks(cplmixVAE_data['num_pruned'][indx])
        ax3.set_xticklabels(cplmixVAE_data['num_pruned'][indx], fontsize=8, rotation=90)
        ax3.legend(loc='upper left')
        ax3.grid(b=True, which='major', linestyle='-')

        plt.show()
        return cplmixVAE_data['num_pruned'], mean_cost, sdt, cplmixVAE_data['con_mean'], cplmixVAE_data['con_min'], indx



def get_SilhScore(x, labels):

    uni_class = np.unique(labels)
    sample_score = silhouette_samples(x, labels)
    sil_score = silhouette_score(x, labels)
    mean_smp_sc = np.zeros(len(uni_class))
    for ic, c in enumerate(uni_class):
        label_ind = np.where(labels == c)[0]
        mean_smp_sc[ic] = np.mean(sample_score[label_ind])

    return mean_smp_sc, sil_score