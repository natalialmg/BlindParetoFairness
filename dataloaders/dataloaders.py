
import sys
sys.path.append("../")

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from general.utils import to_categorical


class TablePandasDataset(Dataset):
    """Pandas dataset.    """

    def __init__(self, pd, cov_list, utility_tag='utility_cat', group_tag=None, weights_tag = None, transform=None):
        """
        Args:
            pd: Pandas dataframe,
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.Y_torch = torch.Tensor(np.vstack(pd[utility_tag].values).astype('float32')) #stacking because they are saved as list of numpy arrays
        self.A_torch = None
        self.W_torch = None
        if group_tag is not None:
            self.A_torch = torch.Tensor(np.vstack(pd[group_tag].values).astype('float32'))
        if weights_tag is not None:
            self.W_torch = torch.Tensor(np.vstack(pd[weights_tag].values).astype('float32'))

        self.X_torch = torch.Tensor(pd[cov_list].to_numpy().astype('float32'))
        self.transform = transform

    def __len__(self):
        return self.Y_torch.shape[0]

    def __getitem__(self, idx):
        # data = self.pd_torch[idx]
        Y = self.Y_torch[idx]
        X = self.X_torch[idx]
        if self.transform:
            X = self.transform(X)
        output = [X,Y]
        if self.A_torch is not None:
            A = self.A_torch[idx]
            output.append(A)
            # return X, Y, A
        if self.W_torch is not None:
            W = self.W_torch[idx]
            output.append(W)
        return output
        # else:
            # return X, Y


def get_dataloaders_tabular(data_pd, utility_tag='utility',
                           cov_tags=['x'], group_tag=None, weights_tag = None, sampler_tag='weights_sampler',
                           sampler_on=False, num_workers = 8, batch_size=32, shuffle=True,transform = None,
                            regression = False,drop_last=False):

    n_utility = data_pd[utility_tag].nunique()
    if not regression:
        data_pd['utility_cat'] = data_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
    else:
        data_pd['utility_cat'] = data_pd[utility_tag]

    if group_tag is not None:
        n_group = data_pd[group_tag].nunique()
        data_pd['group_cat'] = data_pd[group_tag].apply(lambda x: to_categorical(x, num_classes=n_group))
        group_tag = 'group_cat'

    if sampler_on:
        data_weights = torch.DoubleTensor(data_pd[sampler_tag].values)
        data_sampler = torch.utils.data.sampler.WeightedRandomSampler(data_weights, len(data_weights))
        shuffle = False #shuffle mutually exclusive with balance_sampler
    else:
        data_sampler = None

    tabular_dataloader = DataLoader(TablePandasDataset(pd=data_pd,cov_list=cov_tags, utility_tag='utility_cat',
                                  group_tag=group_tag,weights_tag=weights_tag,transform=transform),
                                  batch_size=batch_size,
                                  sampler=data_sampler,shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=True,drop_last=drop_last)

    return tabular_dataloader