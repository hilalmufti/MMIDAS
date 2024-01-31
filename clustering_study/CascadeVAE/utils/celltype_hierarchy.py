import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from utils.analysis_tree_helpers import parse_dend, get_ancestors



def cell_nodes_dict(treeobj, num_cell=132):

    dict = {}
    for i, s in enumerate(treeobj.child):
        if i <=num_cell:
            ancestor = get_ancestors(s, treeobj.child, treeobj.parent)
            dict[s] = ancestor

    return dict

def hierarchy_plot(treeobj, p_cat, unique_types):

    not_used_parents = np.array(['n118', 'n119', 'n120','n121','n122',
                                        'n123','n124', 'n125', 'n126',
                                        'n127', 'n128', 'n129', 'n130',
                                        'n131', 'n132', 'n1', 'n2', 'n3'])

    xx = treeobj.x
    yy = treeobj.y
    yy[np.isnan(yy)] = 0
    col = treeobj.col
    col[~treeobj.isleaf] = '#000000'

    sns.set_style("white")
    fig = plt.figure(figsize=(9, 3))

    for p in treeobj.parent:
        check = 0
        check = np.sum([1 for k in not_used_parents if p==k])
        if check == 0:
            xp = xx[treeobj.child == p]
            yp = yy[treeobj.child == p]
            ch = treeobj.child[treeobj.parent == p]
            for c in ch:
                xc = xx[treeobj.child == c]
                yc = yy[treeobj.child == c]
                plt.plot([xc, xc], [-yc, -yp], color='#BBBBBB')
                plt.plot([xc, xp], [-yp, -yp], color='#BBBBBB')


    for i, s in enumerate(treeobj.child):
        if i < p_cat.shape[0]:
            cel_ty = treeobj.child[treeobj.x == xx[i]][0]
            while not (np.array(unique_types) == cel_ty).any():
                cel_ty = treeobj.get_ancestors(cel_ty)[0]

            cluster_id = unique_types.index(cel_ty)

            plt.plot(xx[i], yy[i], 's', c=col[i], ms=1)
            barplt = plt.bar(xx[i], height=p_cat[cluster_id], width=1,
                             bottom=yy[i] + 0.03, align='center', color=col[i])
            # prob = barplt.get_height()
            # plt.text(barplt.get_x() + barplt.get_width() / 2.,
            #           1.05 * prob, '%s' % nodes[i], rotation=90,
            #           ha='center', va='bottom', fontsize=5)


    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([np.min(xx) - 1, np.max(xx) + 1])
    ax.set_ylim([-0.5, 1.1])
    ax.axis('off')

    return ax, fig


def heatmap_plot(treeobj, cluster_per_cat, unique_types, markSize=1):

    not_used_parents = np.array(['n118', 'n119', 'n120','n121','n122',
                                        'n123','n124', 'n125', 'n126',
                                        'n127', 'n128', 'n129', 'n130',
                                        'n131', 'n132', 'n1', 'n2', 'n3'])

    xx = treeobj.x
    yy = treeobj.y
    yy[np.isnan(yy)] = 0
    col = treeobj.col
    col[~treeobj.isleaf] = '#000000'

    sns.set_style("white")
    fig1 = plt.figure(figsize=(9, 3))

    id = []
    distance = np.zeros((len(unique_types), len(unique_types)))
    for i, s in enumerate(unique_types):
        if i < distance.shape[0]:
            cel_ty = treeobj.child[treeobj.x == xx[
                i]][0]
            while not (np.array(unique_types) == cel_ty).any():
                cel_ty = treeobj.get_ancestors(cel_ty)[0]

            cluster_id = unique_types.index(cel_ty)
            id.append(cluster_id)

            #
            # for j in range(len(unique_types)):
            #     ct = treeobj.child[treeobj.x == xx[j]][0]
            #
            #     while not (np.array(unique_types) == ct).any():
            #         ct = treeobj.get_ancestors(ct)[0]
            #     id_j = unique_types.index(ct)
            #
            #     dz = np.squeeze(mean_prob[cluster_id, :]) - \
            #          np.squeeze(mean_prob[id_j, :])
            #     distance[i, j] = np.linalg.norm(dz) ** 2
            # add colorful markers for cell types
            # plt.plot(xx[i], yy[i]-k, 's', c=col[i], ms=markSize)

    y_scale = 100
    k_y = 1
    k_x = .5

    # for parent in np.unique(treeobj.parent):
    #     # Get position of the parent node:
    #     p_ind = np.flatnonzero(treeobj.child == parent)
    #     if p_ind.size == 0:  # Enters here for any root node
    #         p_ind = np.flatnonzero(treeobj.parent == parent)
    #         xp = treeobj.x[p_ind]
    #         yp = y_scale * np.max(treeobj.y)
    #     else:
    #         xp = treeobj.x[p_ind]
    #         yp = y_scale * treeobj.y[p_ind]
    #
    #     all_c_inds = np.flatnonzero(np.isin(treeobj.parent, parent))
    #     for c_ind in all_c_inds:
    #         xc = treeobj.x[c_ind]
    #         yc = y_scale * treeobj.y[c_ind]
    #         plt.plot([-yc-k_y, -yp-k_y], [xc + k_x, xc + k_x], color='#BBBBBB')
    #         plt.plot([-yp-k_y, -yp-k_y], [xc + k_x, xp + k_x], color='#BBBBBB')


    # for i, s in enumerate(treeobj.child):
    #     if i < distance.shape[0]:
    #         plt.plot(yy[i]-k, xx[i], 's', c=col[i], ms=markSize)

    # sns.heatmap(distance, xticklabels=False,
    #             yticklabels=False, cmap ='gist_gray', vmin=0, vmax=2)  #vlag
    #
    # # fig1.tight_layout()
    # # fig.subplots_adjust(bottom=0.2)
    # ax1 = plt.gca()
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax1.set_xlim([np.min(xx) - 50, np.max(xx) + 1])
    # ax1.set_ylim([-35, 118])

    sns.set_style("white")
    x_b = 0.
    vmax = 1
    linewidth = 2
    y_scale = 5
    fig2 = plt.figure(figsize=(10, 6))
    if distance.shape[0] > 20:
        #fig2 = plt.figure(figsize=(10, 9))
        linewidth = 1.5
        width =.5
        y_scale = 15
    if distance.shape[0] > 50:
        #fig2 = plt.figure(figsize=(10, 8))
        linewidth = 1.2
        width = .8
        y_scale = 35
    if distance.shape[0] > 80:
        #fig2 = plt.figure(figsize=(10, 7))
        linewidth = 1.
        width = .8
        y_scale = 70
        vmax=.5
    if distance.shape[0] > 100:
        #fig2 = plt.figure(figsize=(10, 6))
        x_b = -.25
        vmax=.5
        width = .8
        linewidth = .8

    k = 1
    fig2 = plt.figure(figsize=(8, 6))

    # for parent in np.unique(treeobj.parent):
    #     # Get position of the parent node:
    #     p_ind = np.flatnonzero(treeobj.child == parent)
    #     if p_ind.size == 0:  # Enters here for any root node
    #         p_ind = np.flatnonzero(treeobj.parent == parent)
    #         xp = treeobj.x[p_ind]
    #         yp = y_scale * np.max(treeobj.y)
    #     else:
    #         xp = treeobj.x[p_ind]
    #         yp = y_scale * treeobj.y[p_ind]
    #
    #     all_c_inds = np.flatnonzero(np.isin(treeobj.parent, parent))
    #     for c_ind in all_c_inds:
    #         xc = treeobj.x[c_ind]
    #         yc = y_scale * treeobj.y[c_ind]
    #         plt.plot([-yc - k_y, -yp - k_y], [xc + k_x, xc + k_x],
    #                  color='#BBBBBB', linewidth=linewidth)
    #         plt.plot([-yp - k_y, -yp - k_y], [xc + k_x, xp + k_x],
    #                  color='#BBBBBB', linewidth=linewidth)

    # for i, s in enumerate(treeobj.child):
    #     if i < distance.shape[0]:
    #         plt.plot(yy[i] - k_y, xx[i] + k_x, 's', c=col[i], ms=markSize)

    tmp = np.squeeze(cluster_per_cat)
    tmp = tmp[id, :]
    row_ind, col_ind = linear_sum_assignment(1 - tmp)
    sns.set(font_scale=1.5)
    axx = sns.heatmap(tmp[:, col_ind], xticklabels=range(1, tmp.shape[-1]+1),
                yticklabels=False, vmin=0, vmax=1,
                cbar_kws={"shrink": 1}, annot_kws={"size": 18}) #"shrink": 1
    axx.invert_yaxis()
    #'cividis'

    # if distance[0].shape[0] > 80:
    #     plt.bar(np.linspace(0, distance.shape[0], distance.shape[0]) +x_b,
    #             distance.shape[0]/6 * np.diag(-tmp[:,col_ind]), width=width,
    #             color='black')


    fig2.tight_layout()
    # fig.subplots_adjust(bottom=0.2)
    ax2 = plt.gca()
    ax2.set_xticks([])
    ax2.set_yticks([])
    # ax2.set_xlim([np.min(xx) - distance.shape[0]/2, cluster_per_cat.shape[-1]
    #               + 1])
    # ax2.set_ylim([-distance.shape[0]/6, distance.shape[0]+1])
    fig2.tight_layout()

    return ax2, fig2

def dent_plot(treeobj, cluster_per_cat):

    not_used_parents = np.array(['n118', 'n119', 'n120','n121','n122',
                                        'n123','n124', 'n125', 'n126',
                                        'n127', 'n128', 'n129', 'n130',
                                        'n131', 'n132', 'n1', 'n2', 'n3'])

    xx = treeobj.x
    yy = treeobj.y
    yy[np.isnan(yy)] = 0
    col = treeobj.col
    col[~treeobj.isleaf] = '#000000'

    sns.set_style("white")
    fig1 = plt.figure(figsize=(9, 3))

    y_scale = 100
    k_y = 1
    k_x = .5

    sns.set_style("white")
    x_b = 0.
    vmax = 1
    fig = plt.figure(figsize=(8, 6))
    if cluster_per_cat.shape[0] > 20:
        #fig2 = plt.figure(figsize=(10, 9))
        linewidth = 1.5
        width =.5
        y_scale = 15
    if cluster_per_cat.shape[0] > 50:
        #fig2 = plt.figure(figsize=(10, 8))
        linewidth = 1.2
        width = .8
        y_scale = 35
    if cluster_per_cat.shape[0] > 80:
        #fig2 = plt.figure(figsize=(10, 7))
        linewidth = 1.
        width = .8
        y_scale = 70
        vmax=.5
    if cluster_per_cat.shape[0] > 100:
        #fig2 = plt.figure(figsize=(10, 6))
        x_b = .5
        vmax=.5
        width = 1
        linewidth = .8

    k = 1

    # for parent in np.unique(treeobj.parent):
    #     # Get position of the parent node:
    #     p_ind = np.flatnonzero(treeobj.child == parent)
    #     if p_ind.size == 0:  # Enters here for any root node
    #         p_ind = np.flatnonzero(treeobj.parent == parent)
    #         xp = treeobj.x[p_ind]
    #         yp = y_scale * np.max(treeobj.y)
    #     else:
    #         xp = treeobj.x[p_ind]
    #         yp = y_scale * treeobj.y[p_ind]
    #
    #     all_c_inds = np.flatnonzero(np.isin(treeobj.parent, parent))
    #     for c_ind in all_c_inds:
    #         xc = treeobj.x[c_ind]
    #         yc = y_scale * treeobj.y[c_ind]
    #         plt.plot([-yc - k_y, -yp - k_y], [xc + k_x, xc + k_x],
    #                  color='#BBBBBB', linewidth=linewidth)
    #         plt.plot([-yp - k_y, -yp - k_y], [xc + k_x, xp + k_x],
    #                  color='#BBBBBB', linewidth=linewidth)

    # for i, s in enumerate(treeobj.child):
    #     if i < distance.shape[0]:
    #         plt.plot(yy[i] - k_y, xx[i] + k_x, 's', c=col[i], ms=markSize)

    sns.set(font_scale=1.5)
    axx = sns.heatmap(cluster_per_cat,
                      xticklabels=range(1, cluster_per_cat.shape[-1]+1),
                yticklabels=False, vmin=0, vmax=1,
                cbar_kws={"shrink": 1}, annot_kws={"size": 18})

    axx.invert_yaxis()

    # if distance[0].shape[0] > 50:
    #     plt.bar(np.linspace(0, distance.shape[0], distance.shape[0]) + x_b,
    #             distance.shape[0]/6 * np.diag(-tmp[:,col_ind]), width=width,
    #             color='darkblue')


    # fig.subplots_adjust(bottom=0.2)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlim([np.min(xx) - cluster_per_cat.shape[0]/2, cluster_per_cat.shape[-1]
    #               + 1])
    # ax.set_ylim([-cluster_per_cat.shape[0]/6, cluster_per_cat.shape[0]+1])
    ax.set_ylabel('Cell Types', fontsize=20)
    ax.set_xlabel('Merged categories', fontsize=20)
    fig.tight_layout()

    return ax, fig