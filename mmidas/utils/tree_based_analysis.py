import matplotlib.pyplot as plt
from utils.analysis_cells_tree import HTree, do_merges
from matplotlib.backends.backend_pdf import PdfPages
#from cplmix.utils.celltype_hierarchy import hierarchy_plot, cell_nodes_dict, heatmap_plot
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

resolution = 600


def get_merged_types(htree_file, cells_labels, num_classes=0, ref_leaf=[], node='n4'):
    # get the tree
    htree = HTree(htree_file=htree_file)
    htree.parent = np.array([c.strip() for c in htree.parent])
    htree.child = np.array([c.strip() for c in htree.child])

    # get a subtree according the to the given node:
    subtree = htree.get_subtree(node=node)
    if len(ref_leaf) > 0:
        ref_leaf = np.array(ref_leaf)
        in_idx = np.array([(ref_leaf == c).any() for c in subtree.child[subtree.isleaf]])
        subtree.child = np.concatenate((subtree.child[subtree.isleaf][in_idx], subtree.child[~subtree.isleaf]))
        subtree.parent = np.concatenate((subtree.parent[subtree.isleaf][in_idx], subtree.parent[~subtree.isleaf]))
        subtree.col = np.concatenate((subtree.col[subtree.isleaf][in_idx], subtree.col[~subtree.isleaf]))
        subtree.x = np.concatenate((subtree.x[subtree.isleaf][in_idx], subtree.x[~subtree.isleaf]))
        subtree.y = np.concatenate((subtree.y[subtree.isleaf][in_idx], subtree.y[~subtree.isleaf]))
        subtree.isleaf = np.concatenate((subtree.isleaf[subtree.isleaf][in_idx], subtree.isleaf[~subtree.isleaf]))

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

    #Plot updated tree:
    kept_subtree = HTree(htree_df=kept_subtree_df)

    kept_subtree_df['isleaf'].loc[
        kept_subtree_df['child'].isin(kept_leaf_nodes)] = True
    kept_subtree_df['y'].loc[
        kept_subtree_df['child'].isin(kept_leaf_nodes)] = 0.0
    mod_subtree = HTree(htree_df=kept_subtree_df)


    mod_subtree.update_layout()


    return merged_cells_labels, mod_subtree, subtree
