import matplotlib.pyplot as plt
from modules.cellType.analysis_cells_tree import HTree, do_merges
from matplotlib.backends.backend_pdf import PdfPages
from modules.cellType.celltype_hierarchy import hierarchy_plot, \
    cell_nodes_dict, \
    heatmap_plot
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

resolution = 600

def cell_category_hist(cluster_per_cat,
                       unique_test_class_label,
                       unique_types,
                       mean_prob,
                       class_name,
                       htree,
                       folder,
                       cpl_flag,
                       n_arm=2,
                       single_file=True):
    """
    Generates figures according to the trained coupled or single mixVAE model
    using the cell types' taxonomy tree.

    input args
        cluster_per_cat: the probabilistic contribution matrix of categories
                         for each cell types.
        unique_test_class_label: list of all cell IDs in the original dataset.
        mean_prob: the average of the histogram for the categorical RV.
        class_name: including cell IDs for all samples used in the training.
        htree_file: an excel file including cell taxonomy tree.
        folder: data folder.
        cpl_flag: a boolean flag to set the operation on coupled mode.
        n_arm: number of autoencoders, must be int.
        single_file: a boolean variable that allow you to save all cell
                     histogram figures in a single pdf file.
    """

    print('Post processing ...')
    # cell_dict = cell_nodes_dict(htree_file)
    # min_activity = .1
    # nodes = []
    # for cell in unique_types:
    #     nodes.append(set(cell_dict[cell]))

    # treeobj = pd.read_csv(htree_file)
    # ordered_labels = treeobj['label'][np.where(treeobj['leaf'] == True)[0]]
    # colors = treeobj['col'][np.where(treeobj['leaf'] == True)[0]]
    # ordered_indx = np.array([i for i in range(len(ordered_labels)) if
    #                          ordered_labels.values[i] in class_name])
    # ordered_labels = ordered_labels.values[ordered_indx]
    # colors = colors.values[ordered_indx]
    #
    # indx = [np.where(class_name == c)[0][0] for c in ordered_labels]
    num_cat = cluster_per_cat.shape[-1]

    # if single_file:
    #     pdf_pages = PdfPages(folder + '/cells_histogram.pdf')

    if cpl_flag:
        for arm in range(n_arm):
            fig, ax = plt.subplots(1, 1)
            # plot the activity patter of each cluster for each cell type
            im = ax.pcolormesh(np.squeeze(cluster_per_cat[arm, :, :]))
            ax.set_aspect(aspect=0.2)
            ax.set_ylabel('Cell Types')
            ax.set_xlabel('Categories')
            # ax.set_title('average categories prob. across all types')
            fig.colorbar(im, ax=ax)
            im.set_clim(0, 1)
            ax.set_xticks(range(num_cat))
            ax.set_yticks(range(len(unique_test_class_label)))
            ax.tick_params(labelsize=2)
            # plt.savefig(folder + '/cells_vs_categories_arm_' + str(arm)
            #             +'.pdf',
            #             dpi=resolution)

            distance = np.zeros((len(unique_test_class_label),
                                 len(unique_test_class_label)))
            for l_1 in unique_test_class_label:
                l_1 = int(l_1) - 1
                for l_2 in unique_test_class_label:
                    l_2 = int(l_2) - 1
                    dz = np.squeeze(mean_prob[0, l_1, :]) - \
                         np.squeeze(mean_prob[0, l_2, :])
                    distance[l_1, l_2] = np.linalg.norm(dz)**2
            fig, ax = plt.subplots(1, 1)
            # plot the activity patter of each cluster for each cell type
            im = ax.pcolormesh(distance)
            fig.colorbar(im, ax=ax)
            im.set_clim(0, 2)
            ax.set_ylabel('Cell Types')
            ax.set_xlabel('Cell Types')

            # ax.set_title('L2 norm between average categorical dist.')
            ax.set_xticks(range(len(unique_test_class_label)))
            ax.set_yticks(range(len(unique_test_class_label)))
            ax.tick_params(labelsize=2)
            # plt.savefig(folder + '/distance_across_cells_arm_' + str(arm) +
            #             '.pdf',
            #             dpi=resolution)
            plt.close("all")
            # heatmap = sns.heatmap(distance, cmap='YlGnBu', xticklabels=False,
            #                       yticklabels=False, vmin=0, vmax=2)
            ax, fig = heatmap_plot(htree, np.squeeze(
                cluster_per_cat[arm,:,:]),  unique_types)
            # ax1.set_ylabel('Cells', fontsize=14)
            # ax1.set_xlabel('Cells', fontsize=14)
            # ax1.set_title('Euclidean distance between clusters', fontsize=16)
            # fig1.set_size_inches(15, 10)
            # fig1.savefig(folder + '/distance_across_cells_arm_' + str(
            #     arm) + '.png', dpi=resolution)

            ax.set_ylabel('Cell Types', fontsize=20)
            ax.set_xlabel('Categories', fontsize=20)
            # ax2.set_title('Average categories probabilities across all cell '
            #               'types', fontsize=14)
            fig.tight_layout()
            fig.savefig(folder + '/cells_vs_categories_arm_' + str(arm) +
                        '_K_' + str(len(unique_types)) + '.png', dpi=resolution)

        # for c in range(num_cat):
        #
        #     ax, fig = hierarchy_plot(htree, cluster_per_cat[0, :, c],
        #                              unique_types)
        #     ax.set_title('Probability Values for Category ' + str(c + 1))
        #
        #     if single_file:
        #         pdf_pages.savefig(fig)
        #     else:
        #         # mng = plt.get_current_fig_manager()
        #         # mng.window.showMaximized()
        #         # plt.tight_layout()
        #         if np.sum(cluster_per_cat[0, :, c]) > 0:
        #             plt.savefig(folder + '/cells_histogram_K' + str(c + 1) +
        #                 '.png', dpi=resolution)
        #
        #     plt.close("all")
        #
        # if single_file:
        #     pdf_pages.close()

    else:
        fig, ax = plt.subplots(1, 1)
        # plot the activity patter of each cluster for each cell type
        im = ax.pcolormesh(np.squeeze(cluster_per_cat))
        ax.set_aspect(aspect=0.2)
        ax.set_ylabel('Cell Types')
        ax.set_xlabel('Categories')
        # ax.set_title('average categories prob. across all types')
        fig.colorbar(im, ax=ax)
        im.set_clim(0, 1)
        ax.set_xticks(range(num_cat))
        ax.set_yticks(range(len(unique_test_class_label)))
        ax.tick_params(labelsize=2)
        plt.savefig(folder + '/cells_vs_categories.pdf', dpi=resolution)

        distance = np.zeros((len(unique_test_class_label),
                             len(unique_test_class_label)))
        for l_1 in unique_test_class_label:
            l_1 = int(l_1) - 1
            for l_2 in unique_test_class_label:
                l_2 = int(l_2) - 1
                dz = np.squeeze(mean_prob[l_1, :]) - \
                     np.squeeze(mean_prob[l_2, :])
                distance[l_1, l_2] = np.linalg.norm(dz) ** 2
        fig, ax = plt.subplots(1, 1)
        # plot the activity patter of each cluster for each cell type
        im = ax.pcolormesh(distance)
        fig.colorbar(im, ax=ax)
        # im.set_clim(0, 2)
        ax.set_ylabel('Cell Types')
        ax.set_xlabel('Cell Types')

        # ax.set_title('L2 norm between average categorical dist.')
        ax.set_xticks(range(len(unique_test_class_label)))
        ax.set_yticks(range(len(unique_test_class_label)))
        ax.tick_params(labelsize=2)
        # plt.savefig(folder + '/distance_across_cells.pdf', dpi=resolution)
        plt.close("all")
        # heatmap = sns.heatmap(distance, cmap='YlGnBu', xticklabels=False,
        #                       yticklabels=False, vmin=0, vmax=2)

        ax1, fig1, ax2, fig2 = heatmap_plot(htree, cluster_per_cat,
                                            mean_prob, unique_types)
        ax1.set_ylabel('Cells', fontsize=14)
        ax1.set_xlabel('Cells', fontsize=14)
        ax1.set_title('Euclidean distance between clusters', fontsize=16)
        fig1.set_size_inches(15, 10)
        # fig1.savefig(folder + '/distance_across_cells_2.pdf', dpi=resolution)

        ax2.set_ylabel('Cells', fontsize=14)
        ax2.set_xlabel('Categories', fontsize=14)
        ax2.set_title('Average categories probabilities across all cell '
                      'types', fontsize=14)
        fig2.savefig(folder + '/cells_vs_categories_2.pdf', dpi=resolution)

        # for c in range(num_cat):
        #     ax, fig = hierarchy_plot(htree, cluster_per_cat[:, c],
        #                              unique_types)
        #     ax.set_title('Probability Values for Category ' + str(c + 1))
        #
        #     if single_file:
        #         pdf_pages.savefig(fig)
        #     else:
        #         if np.sum(cluster_per_cat[:, c]) > 0:
        #             plt.savefig(folder + '/cells_histogram_K' + str(c + 1) +
        #                         '.png', dpi=resolution)
        #     plt.close("all")

        # if single_file:
        #     pdf_pages.close()


def init_category_hist(unique_types, class_name, htree_file):
    """
    Generates figures according to the trained coupled or single mixVAE model
    using the cell types' taxonomy tree.

    input args
        cluster_per_cat: the probabilistic contribution matrix of categories
                         for each cell types.
        unique_test_class_label: list of all cell IDs in the original dataset.
        mean_prob: the average of the histogram for the categorical RV.
        class_name: including cell IDs for all samples used in the training.
        htree_file: an excel file including cell taxonomy tree.
        folder: data folder.
        cpl_flag: a boolean flag to set the operation on coupled mode.
        n_arm: number of autoencoders, must be int.
        single_file: a boolean variable that allow you to save all cell
                     histogram figures in a single pdf file.
    """

    cell_dict = cell_nodes_dict(htree_file)
    nodes = []
    for cell in unique_types:
        nodes.append(set(cell_dict[cell]))

    treeobj = pd.read_csv(htree_file)
    ordered_labels = treeobj['label'][np.where(treeobj['leaf'] == True)[0]]
    colors = treeobj['col'][np.where(treeobj['leaf'] == True)[0]]
    ordered_indx = np.array([i for i in range(len(ordered_labels)) if
                             ordered_labels.values[i] in class_name])
    ordered_labels = ordered_labels.values[ordered_indx]
    indx = [np.where(class_name == c)[0][0] for c in ordered_labels]

    return indx


def training_category_hist(cluster_per_cat,
                           unique_test_class_label,
                           unique_types,
                           mean_prob,
                           htree_file,
                           folder,
                           indx,
                           epoch):

    res = 300
    num_cat = cluster_per_cat.shape[-1]
    distance = np.zeros((len(unique_test_class_label),
                         len(unique_test_class_label)))
    for l_1 in unique_test_class_label:
        l_1 = int(l_1) - 1
        for l_2 in unique_test_class_label:
            l_2 = int(l_2) - 1
            dz = np.squeeze(mean_prob[l_1, :]) - \
                 np.squeeze(mean_prob[l_2, :])
            distance[l_1, l_2] = np.linalg.norm(dz) ** 2

    ax1, fig1, ax2, fig2 = heatmap_plot(htree_file, cluster_per_cat,
                                        mean_prob, unique_types)
    ax1.set_ylabel('Cells', fontsize=14)
    ax1.set_xlabel('Cells', fontsize=14)
    ax1.set_title('Euclidean distance between clusters', fontsize=16)
    fig1.set_size_inches(15, 10)
    fig1.savefig(folder + '/distance_frames/distance_across_cells_epoch_'
                        + str(epoch) + '.png', dpi=res)

    ax2.set_ylabel('Cells', fontsize=14)
    ax2.set_xlabel('Categories', fontsize=14)
    ax2.set_title('Average categories probabilities across all cell '
                  'types', fontsize=14)
    fig2.savefig(folder + '/cell_cat_frames/cells_vs_categories_epoch_'
                        + str(epoch) + '.png', dpi=res)

    for c in range(num_cat):
        active_cells = np.where(cluster_per_cat[indx,c] > 1/num_cat)[0]
        # intersect_nodes = set.intersection(*[nodes[ac] for ac in
        #                                      active_cells])
        ax, fig = hierarchy_plot(htree_file, cluster_per_cat[:, c],
                                 unique_types)
        ax.set_title('Probability Values for Category ' + str(c + 1))
        plt.savefig(folder + '/hist_frames/cells_histogram_K' +
                    str(c + 1) + '_epoch_' + str(epoch) + '.png', dpi=res)
        plt.close("all")




def corr_analysis(state, cell):

        n_gene = cell.shape[-1]
        all_corr, all_geneID = [], []
        for s in range(state.shape[-1]):
            # compute cross correlation using Pearson correction coefficient
            cor_coef, p_val = np.zeros(n_gene), np.zeros(n_gene)
            for g in range(n_gene):
                if np.max(cell[:, g]) > 0:
                    zind = np.where(cell[:, g] > 0)
                    if len(zind[0]) > 4:
                        cor_coef[g], p_val[g] = \
                            stats.pearsonr(state[zind[0], s],
                                           cell[zind[0], g])
                    else:
                        cor_coef[g], p_val[g] = 0, 0
                else:
                    cor_coef[g], p_val[g] = 0, 0

            g_id = np.argsort(np.abs(cor_coef))
            # gene.append(dataset['gene_id'][g_id[-10:]])
            # max_corr.append(cor_coef[g_id])
            all_corr.append(np.sort(np.abs(cor_coef)))
            all_geneID.append(g_id)

            # create a linear regression model
            # zind = np.where(expression[:, g_id] > 0)
            # x = s_val[zind[0], s]
            # y = expression[zind[0], g_id]
            # model = LinearRegression()
            # model.fit(np.expand_dims(x, -1), np.expand_dims(y, -1))
            #
            # # predict y from the data
            # x_new = np.linspace(np.min(x)-.2, np.max(x)+.2, 100)
            # y_new = model.predict(x_new[:, np.newaxis])
            #
            # # plot the results
            # ax = plt.axes()
            # ax.scatter(x, y, alpha=0.3, s=15, c='black')
            # ax.plot(x_new, y_new, c='black')
            # ax.axis('scaled')
            # ax.set_ylim([np.min(y)-0.2, np.max(y)+0.2])
            # ax.set_xlabel('S conditioning on Z=' + str(cat+1),
            #               fontsize=8)
            # ax.set_ylabel('Gene Expression for ' + gene[-1], fontsize=8)
            # ax.set_title('corr. coef. {:.2f}'.format(max_corr[-1]))
            # plt.savefig(folder + '/state_analysis/state_' + str(s) +
            #             '_gene_corr_K' + str(
            #     cat+1) + '_g_' + gene[-1] +
            #             '.png', dpi=resolution, bbox_inches='tight')
            # plt.close('all')

        return all_corr, all_geneID


def get_merged_types(htree_file, cells_labels, num_classes=0, node='n4'):
    # get the tree
    htree = HTree(htree_file=htree_file)

    # get a subtree according the to the given node:
    subtree = htree.get_subtree(node=node)

    # get a list of merges to carry out, sorted by the depth
    L = subtree.get_mergeseq()

    if num_classes == 0:
        go = len(L)
    else:
        go = num_classes

    merged_cells_labels = do_merges(labels=cells_labels,
                                    list_changes=L,
                                    n_merges=(go-1), verbose=False)

    unique_merged_cells_labels = do_merges(labels=subtree.child[subtree.isleaf],
                                    list_changes=L,
                                    n_merges=(go-1), verbose=False)

    # Obtain all relevant ancestor nodes:
    kept_leaf_nodes = list(set(unique_merged_cells_labels.tolist()))
    kept_tree_nodes = []
    for node in kept_leaf_nodes:
        kept_tree_nodes.extend(subtree.get_ancestors(node))
        kept_tree_nodes.extend([node])

    kept_subtree_df = subtree.obj2df()
    kept_subtree_df = kept_subtree_df[kept_subtree_df['child'].isin(kept_tree_nodes)]

    # Plot updated tree:
    kept_subtree = HTree(htree_df=kept_subtree_df)

    kept_subtree_df['isleaf'].loc[
        kept_subtree_df['child'].isin(kept_leaf_nodes)] = True
    kept_subtree_df['y'].loc[
        kept_subtree_df['child'].isin(kept_leaf_nodes)] = 0.0
    mod_subtree = HTree(htree_df=kept_subtree_df)


    mod_subtree.update_layout()


    return merged_cells_labels, mod_subtree, subtree


