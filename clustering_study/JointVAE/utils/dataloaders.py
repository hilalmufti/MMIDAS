import glob, torch
import numpy as np
import anndata
from torch.utils.data import Dataset, DataLoader, sampler, TensorDataset
from torchvision import datasets, transforms
from sklearn.utils import shuffle


def dataIO_cpm(dataset, cvset, n_genes, train_per, val_per, test_per):
    """
    Shuffles and splits the original dataset for train and validation.

    input args
        dataset: the entire dataset used for training and validation.
        cvset: seed of random generator.
        n_genes: number of genes (features) used for training.
        train_size: size of the training data component.
        val_size: size of the validation data component.
        test_size: size of the test data component.

    return
        train_cpm: training data component.
        val_cpm: validation data component.
        test_cpm: test data component.
        train_ind: index of training data component in the original dataset.
        val_ind: index of validation data component in the original dataset.
        test_ind: index of test data component in the original dataset.
    """
    label_list = np.unique(dataset['class_label'])
    data_size = len(dataset['class_label'])
    dataset['classID'] = -np.ones(data_size)
    data_per_class = np.zeros(len(label_list))
    id = 0
    for label in label_list:
        dataset['classID'][np.where(dataset['class_label'] == label)] = id
        data_per_class[id] = np.int(np.sum(dataset['class_label'] == label))
        id += 1

    data_cpm = dataset['log1p'][:, :n_genes]
    label_cpm = dataset['classID']
    train_cpm = np.empty((0, n_genes))
    val_cpm = np.empty((0, n_genes))
    test_cpm = np.empty((0, n_genes))
    train_label = np.empty(0)
    val_label = np.empty(0)
    test_label = np.empty(0)
    train_ind = np.empty(0)
    val_ind = np.empty(0)
    test_ind = np.empty(0)
    id = 0
    for label in label_list:
        indx = np.where(label_cpm == id)
        class_data = np.squeeze(data_cpm[indx, :], axis=0)
        train_size = np.int(train_per * data_per_class[id])
        test_size = np.int(test_per * data_per_class[id])
        val_size = np.int(data_per_class[id] - train_size - test_size)
        train_cpm = np.append(train_cpm, class_data[:train_size, :], axis=0)
        # train_cpm.append(class_data[:train_size,:])
        val_cpm = np.append(val_cpm, class_data[
                                     train_size:train_size + val_size, :],
                            axis=0)
        # val_cpm.append(class_data[train_size:train_size+val_size,:])
        test_cpm = np.append(test_cpm, class_data[
                                       train_size + val_size:, :], axis=0)
        # test_cpm.append(class_data[train_size+val_size:,:])
        train_label = np.append(train_label, id * np.ones(train_size), axis=0)
        val_label = np.append(val_label, id * np.ones(val_size), axis=0)
        test_label = np.append(test_label, id * np.ones(test_size), axis=0)
        train_ind = np.append(train_ind, indx[0][:train_size], axis=0)
        val_ind = np.append(val_ind, indx[0][train_size:train_size + val_size],
                            axis=0)
        test_ind = np.append(test_ind, indx[0][train_size + val_size:], axis=0)
        id += 1
    # shuffle train, validation, and test sets
    train_cpm, train_shuff_ind = shuffle(train_cpm, range(train_cpm.shape[0]))
    val_cpm, val_shuff_ind = shuffle(val_cpm, range(val_cpm.shape[0]))
    test_cpm, test_shuff_ind = shuffle(test_cpm, range(test_cpm.shape[0]))
    train_label = train_label[train_shuff_ind].astype(int)
    val_label = val_label[val_shuff_ind].astype(int)
    test_label = test_label[test_shuff_ind].astype(int)
    train_ind = train_ind[train_shuff_ind].astype(int)
    val_ind = val_ind[val_shuff_ind].astype(int)
    test_ind = test_ind[test_shuff_ind].astype(int)
    return data_cpm, train_cpm, val_cpm, test_cpm, train_ind, val_ind, \
           test_ind, \
           train_label, val_label, test_label


def gene_dataloader(dataset, n_features, batch_size):
    data_cpm, train_cpm, val_cpm, test_cpm, train_ind, val_ind, test_ind, \
    train_label, val_label, test_label = dataIO_cpm(dataset,
                                                    cvset=0,
                                                    n_genes=n_features,
                                                    train_per=.9,
                                                    val_per=.05,
                                                    test_per=.05)
    train_cpm_torch = torch.FloatTensor(train_cpm)
    train_ind_torch = torch.FloatTensor(train_ind)
    train_data = TensorDataset(train_cpm_torch, train_ind_torch)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_cpm_torch = torch.FloatTensor(val_cpm)
    val_ind_torch = torch.FloatTensor(val_ind)
    validation_data = TensorDataset(val_cpm_torch, val_ind_torch)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    test_cpm_torch = torch.FloatTensor(test_cpm)
    test_ind_torch = torch.FloatTensor(test_ind)
    test_data = TensorDataset(test_cpm_torch, test_ind_torch)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    data_cpm_troch = torch.FloatTensor(data_cpm)
    all_ind_torch = torch.FloatTensor(range(len(data_cpm)))
    all_data = TensorDataset(data_cpm_troch, all_ind_torch)
    alldata_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, validation_loader, test_loader, alldata_loader


def get_indices(dataset, class_name):
    rest_indices = []
    class_indices = []
    k = 0
    for c in class_name:
        selected_indices = []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] == c:
                selected_indices.append(i)

            if k == 0:
                n_index = 0
                n_index = sum([1 for cl in class_name if dataset.targets[i] == cl])
                if n_index == 0:
                    rest_indices.append(i)

        class_indices.append(selected_indices)
        k += 1

    return class_indices, rest_indices

def get_mnist_dataloaders(batch_size=128, selected_digit=[0], digit_rm_per=[],
                          path_to_data='../data'):
    """MNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)

    percent = np.zeros(10)
    all_ind_train = []
    all_ind_test = []

    sel_ind_train, rst_ind_train = get_indices(train_data, selected_digit)
    sel_ind_test, rst_ind_test = get_indices(test_data, selected_digit)

    if len(digit_rm_per) > 0:
        for i in range(len(selected_digit)):
            remain_num = int((1 - digit_rm_per[i]) * len(sel_ind_train[i]))
            all_ind_train += sel_ind_train[i][:remain_num]
            remain_num = int((1 - digit_rm_per[i]) * len(sel_ind_test[i]))
            all_ind_test += sel_ind_test[i][:remain_num]
            percent[selected_digit[i]] = digit_rm_per[i]

        all_ind_train += rst_ind_train
        all_ind_test += rst_ind_test

        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  sampler=sampler.SubsetRandomSampler(
                                      all_ind_train),
                                  drop_last=True)

        test_loader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 sampler=sampler.SubsetRandomSampler(
                                     all_ind_test))
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    digit_count = []
    i = 0
    for d in np.unique(train_data.train_labels):
        digit_count.append(int((1 - percent[i]) *
                               len(np.where(train_data.train_labels == d)[0])))
        i += 1

    digit_count = np.array(digit_count)/len(train_data)
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(batch_size=128,
                                  path_to_data='../fashion_data'):
    """FashionMNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.FashionMNIST(path_to_data, train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST(path_to_data, train=False,
                                      transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_dsprites_dataloader(batch_size=128,
                            path_to_data='../dsprites-data/dsprites_data.npz'):
    """DSprites dataloader."""
    dsprites_data = DSpritesDataset(path_to_data,
                                    transform=transforms.ToTensor())
    dsprites_loader = DataLoader(dsprites_data, batch_size=batch_size,
                                 shuffle=True)
    return dsprites_loader


def get_chairs_dataloader(batch_size=128,
                          path_to_data='../rendered_chairs_64'):
    """Chairs dataloader. Chairs are center cropped and resized to (64, 64)."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=True)
    return chairs_loader


def get_chairs_test_dataloader(batch_size=62,
                               path_to_data='../rendered_chairs_64_test'):
    """There are 62 pictures of each chair, so get batches of data containing
    one chair per batch."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=False)
    return chairs_loader


def get_celeba_dataloader(batch_size=128, path_to_data='../celeba_64'):
    """CelebA dataloader with (64, 64) images."""
    celeba_data = CelebADataset(path_to_data,
                                transform=transforms.ToTensor())
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=True)
    return celeba_loader


class DSpritesDataset(Dataset):
    """D Sprites dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.imgs = np.load(path_to_data)['imgs'][::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Each image in the dataset has binary values so multiply by 255 to get
        # pixel values
        sample = self.imgs[idx] * 255
        # Add extra dimension to turn shape into (H, W) -> (H, W, C)
        sample = sample.reshape(sample.shape + (1,))

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0


class CelebADataset(Dataset):
    """CelebA dataset with 64 by 64 images."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.img_paths = glob.glob(path_to_data + '/*')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = imread(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0


def load_10x_data(datafile, measure='log1p', n_gene=0, gene_id=[], ref_genes=False, ref_types=False, ann_smt='', ann_10x='', min_num=10):

    key_list = [measure, 'gene_id', 'sample_name', 'cluster_color', 'cluster_order', 'cluster_label', 'class_color',
                'class_order', 'class_label', 'subclass_color', 'subclass_order', 'subclass_label', 'sex_id',
                'donor_sex_label', 'region_color', 'region_id', 'region_label', 'cortical_layer_label']
    adata = anndata.read_h5ad(datafile)
    print('data is loaded.')

    data = dict()
    data[measure] = adata.X
    features = adata.var.keys()
    data['gene_id'] = adata.var[features].values
    anno_key = adata.obs.keys()
    for key in anno_key:
        data[key] = adata.obs[key].values

    if n_gene == 0:
        n_gene = len(data['gene_id'])

    if len(gene_id) > 0:
        gene_idx = [np.where(data['gene_id'] == gg)[0] for gg in gene_id]
        gene_idx = np.concatenate(gene_idx).astype(int)
        data['gene_id'] = data['gene_id'][gene_idx]
        data[measure] = data[measure][:, gene_idx].todense()
        n_gene = len(gene_idx)
    else:
        data['gene_id'] = data['gene_id'][:n_gene]
        data[measure] = data[measure][:, :n_gene].todense()

    print(data[measure].shape)

    all_key = list(data.keys())
    for key in all_key:
        if key not in key_list:
            del data[key]

    # remove cells "CR", "NP PPP", "Ndnf", and "Pax"
    ind = np.where(data['cluster_label'] == '1_CR')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    ind = np.where(data['cluster_label'] == '354_NP PPP')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    ind = np.where(data['cluster_label'] == '22_Ndnf HPF')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    ind = np.where(data['cluster_label'] == '23_Ndnf HPF')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    ind = np.where(data['cluster_label'] == '17_Pax6')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    ind = np.where(data['cluster_label'] == '18_Pax6')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    ind = np.where(data['cluster_label'] == '19_Pax6')[0]
    for key in data:
        if key not in ['gene_id']:
            data[key] = np.delete(data[key], ind, axis=0)

    uniq_clusters = np.unique(data['cluster_label'])

    count = np.zeros(len(uniq_clusters))
    for it, tt in enumerate(uniq_clusters):
        count[it] = sum(data['cluster_label'] == tt)

    uniq_clusters = uniq_clusters[count >= min_num]
    subclass_ind = []
    for tt in uniq_clusters:
        subclass_ind.append(np.array([i for i in range(len(data['cluster_label'])) if tt == data['cluster_label'][i]]))

    subclass_ind = np.concatenate(subclass_ind)
    ref_len = len(data['cluster_label'])

    for k in key_list:
        if k == measure:
            data[k] = np.array(data[k])[subclass_ind, :]
        elif k not in ['gene_id', measure]:
            data[k] = np.array(data[k])[subclass_ind]

    data['cluster_id'] = np.zeros(len(data['cluster_order']))
    for ic, cls_ord in enumerate(np.unique(data['cluster_label'])):
        data['cluster_id'][data['cluster_label'] == cls_ord] = int(ic + 1)

    print(len(np.unique(data['cluster_label'])), data[measure].shape, len(data['gene_id']))

    return data
