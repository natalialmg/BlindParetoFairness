
import os
import pandas as pd
import sys
import numpy as np
sys.path.append("../")
from dataloaders.data_preprocessing import extract_sensitive_columns,group_concatenation,compute_mean_std_tables
from sklearn.model_selection import StratifiedKFold

##### UCI ADULT DATASET #####
dirdata_UCIadult = '/data/MLTdata/uci_adult/dataset_processed/'

def UCIadult_pandas(groups_list=['sex', 'race_bin'], utility='income', norm_std=True, seed=42,split = 1,n_splits=5):
    pd_data_o = pd.read_csv(os.path.join(dirdata_UCIadult, 'bpf_dataset_cat.csv'))

    s_columns = ['race', 'race_bin', 'race_full', 'sex', 'income', 'sample',
                 'native-country']  # remove & extract sensitive columns
    pd_data, s_data_dic = extract_sensitive_columns(s_columns, pd_data_o)
    cov_tags = pd_data.columns.values  # covariance tags

    for col in s_columns: #add all columns again (they will not be included in cov tags)
        if col in pd_data_o.columns:
            pd_data[col] = pd_data_o[col].values

    ## set utility variables and reincorporate samples index
    pd_data['utility'] = s_data_dic[utility].values
    pd_data['sample_index'] = s_data_dic['sample'].values

    ### Train - Test split ###
    strat_list = [] #stratification tags for balance splitting
    for group in groups_list:
        strat_list.append(group)
    strat_list.append('income')

    values_strat, int2tag_strat, tag2int_strat = group_concatenation(s_data_dic, strat_list,
                                                                     tag2int={}, int2tag=[])
    pd_data['strat'] = np.array(int2tag_strat)[values_strat, -1]

    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    X = pd_data.values
    y = np.array(values_strat)
    split_i = 1
    if split < split_i:
        print('SPLIT < 1 setting to : 1')
        split = 1
    if split > n_splits:
        print('SPLIT > n_splits setting to n_splits ')
        split = n_splits
    for train_index, test_index in skf.split(X, y):
        # print('*** split: ', split_i)
        # print("TRAIN:", train_index, "TEST:", test_index)
        # print('% train ', len(train_index) / (len(train_index) + len(test_index)))
        if split_i == split:
            print(' Split : ',split_i, ' is loaded...')
            dataset_list = ['train' if ix in train_index else 'test' for ix in range(len(pd_data))]
            pd_data['dataset'] = dataset_list
            print(np.unique(np.array(dataset_list)[np.array(train_index).astype('int')]),
                  np.unique(np.array(dataset_list)[np.array(test_index).astype('int')]))  # sanity check
        # print()
        split_i += 1

    pd_train = pd_data.loc[pd_data.dataset == 'train']
    pd_test = pd_data.loc[pd_data.dataset == 'test']

    ### Normalize
    if norm_std:
        ignore_tags = ['utility', 'groups']
        pd_full = pd.concat([pd_train, pd_test])
        mean_std_dict = compute_mean_std_tables(pd_full, cov_tags=cov_tags, save_dir=None)
        for c in mean_std_dict.keys():
            if c in ignore_tags:
                print(c)
                continue
            mean, std = mean_std_dict[c]
            if std > 0:
                pd_train[c] = (pd_train[c].values - mean) / std
                pd_test[c] = (pd_test[c].values - mean) / std
            else:
                print('problematic ', c, '; (mean,std) ', mean, std)

    check = all(item in pd_train['sample_index'].values for item in pd_test['sample_index'].values)
    print('------- UCI Adult Dataset processing ---------- ')
    print('utility : ', utility, '; stratification_tags : ', strat_list, '; standarization : ',
          norm_std, '; len(cov_tags) : ',len(cov_tags))
    print('ntrain : ', len(pd_train), ' ; ntest : ', len(pd_test))
    print('check: overlap train and test ? ', check)
    print('')

    return pd_train, pd_test, cov_tags


dirdata_compas = '/data/MLTdata/compas/dataset_processed/'

def Compas_pandas(groups_list=['race_bin','sex'], utility='two_year_recid', norm_std=True, split=1,n_splits=5,
                  seed=42):
    pd_data_o = pd.read_csv(os.path.join(dirdata_compas, 'bpf_dataset_cat.csv'))

    ## before processing remove all rows with nans in utility col
    if ('two_year_recid' == utility) | ('two_year_recid' in groups_list):
        pd_data_o = pd_data_o[np.isfinite(pd_data_o['two_year_recid'])]

    # remove & extract categorical sensitive columns
    s_columns = ['race', 'race_bin', 'sex', 'two_year_recid', 'sample']

    pd_data, s_data_dic = extract_sensitive_columns(s_columns, pd_data_o, s_inc_col=False)
    cov_tags = pd_data.columns.values  # covariance tags
    for col in s_columns: #add all columns again (they will not be included in cov tags)
        if col in pd_data_o.columns:
            pd_data[col] = pd_data_o[col].values

            ## set utility variables and reincorporate samples index
    pd_data['utility'] = s_data_dic[utility].values
    pd_data['sample_index'] = s_data_dic['sample'].values

    ### Train - Test split ###
    strat_list = []  # stratification tags for balance splitting
    for group in groups_list:
        strat_list.append(group)
    strat_list.append(utility)

    values_strat, int2tag_strat, tag2int_strat = group_concatenation(s_data_dic, strat_list,
                                                                     tag2int={}, int2tag=[])
    pd_data['strat'] = np.array(int2tag_strat)[values_strat, -1]

    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    X = pd_data.values
    y = np.array(values_strat)
    split_i = 1
    if split < split_i:
        print('SPLIT < 1 setting to : 1')
        split = 1
    if split > n_splits:
        print('SPLIT > n_splits setting to n_splits ')
        split = n_splits
    for train_index, test_index in skf.split(X, y):
        # print('*** split: ', split_i)
        # print("TRAIN:", train_index, "TEST:", test_index)
        # print('% train ', len(train_index) / (len(train_index) + len(test_index)))
        if split_i == split:
            print(' Split : ', split_i, ' is loaded...')
            dataset_list = ['train' if ix in train_index else 'test' for ix in range(len(pd_data))]
            pd_data['dataset'] = dataset_list
            print(np.unique(np.array(dataset_list)[np.array(train_index).astype('int')]),
                  np.unique(np.array(dataset_list)[np.array(test_index).astype('int')]))  # sanity check
        # print()
        split_i += 1

    pd_train = pd_data.loc[pd_data.dataset == 'train']
    pd_test = pd_data.loc[pd_data.dataset == 'test']

    ### Normalize
    if norm_std:
        ignore_tags = ['utility', 'groups']
        pd_full = pd.concat([pd_train, pd_test])
        mean_std_dict = compute_mean_std_tables(pd_full, cov_tags=cov_tags, save_dir=None)
        for c in mean_std_dict.keys():
            if c in ignore_tags:
                print(c)
                continue
            mean, std = mean_std_dict[c]
            if std > 0:
                pd_train[c] = (pd_train[c].values - mean) / std
                pd_test[c] = (pd_test[c].values - mean) / std
            else:
                print('problematic ', c, '; (mean,std) ', mean, std)

    check = all(item in pd_train['sample_index'].values for item in pd_test['sample_index'].values)
    print('------- Compas Dataset processing ---------- ')
    print('utility : ', utility, '; stratification_tags : ', strat_list, '; standarization : ',
          norm_std, '; len(cov_tags) : ', len(cov_tags))
    print('ntrain : ', len(pd_train), ' ; ntest : ', len(pd_test))
    print('check: overlap train and test ? ', check)
    print('')

    return pd_train, pd_test, cov_tags


##### Lawschool DATASET #####

dirdata_lawschool = '/data/MLTdata/law_school/dataset_processed/'

def lawschool_pandas(groups_list=['sex','race_bin'], utility='pass_bar', norm_std=True, split = 1,
                     seed=42,n_splits = 5):

    pd_data_o = pd.read_csv(os.path.join(dirdata_lawschool, 'bpf_dataset_cat.csv'))

    ## before processing remove all rows with nans in utility col
    if ('pass_bar' == utility) | ('pass_bar' in groups_list):
        pd_data_o = pd_data_o[np.isfinite(pd_data_o['pass_bar'])]

    if ('zfygpa' == utility) | ('zfygpa' in groups_list):
        pd_data_o = pd_data_o[np.isfinite(pd_data_o['zfygpa'])]


    # remove & extract categorical sensitive columns
    s_columns = ['race', 'race_bin', 'sex', 'zfygpa', 'pass_bar', 'sample']
    if 'fam_inc' in groups_list:
        s_columns.append('fam_inc')
    else:
        pd_data_o = pd_data_o[pd_data_o['fam_inc'] != '?']  # drop '?' because fam_inc will be an integer input
        pd_data_o['fam_inc'] = pd_data_o['fam_inc'].values.astype('float')

    pd_data, s_data_dic = extract_sensitive_columns(s_columns, pd_data_o,s_inc_col=True)
    cov_tags = pd_data.columns.values  # covariance tags
    for col in s_columns: #add all columns again (they will not be included in cov tags)
        if col in pd_data_o.columns:
            pd_data[col] = pd_data_o[col].values

    print(groups_list, s_data_dic.keys())

    ## set utility variables and reincorporate samples index
    pd_data['utility'] = s_data_dic[utility].values
    pd_data['sample_index'] = s_data_dic['sample'].values


    ### Group Tags

    strat_list = []

    if 'pass_bar' == utility :
        strat_list.append('pass_bar')
    # if 'fam_inc' not in groups_list:
        # strat_list.append('fam_inc')
    strat_list.extend(groups_list)

    values_strat, int2tag_strat, tag2int_strat = group_concatenation(s_data_dic, strat_list,
                                                                     tag2int={}, int2tag=[])
    pd_data['strat'] = np.array(int2tag_strat)[values_strat, -1]

    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    X = pd_data.values
    y = np.array(values_strat)
    split_i = 1
    if split < split_i:
        print('SPLIT < 1 setting to : 1')
        split = 1
    if split > n_splits:
        print('SPLIT > n_splits setting to n_splits ')
        split = n_splits
    for train_index, test_index in skf.split(X, y):
        # print('*** split: ', split_i)
        # print("TRAIN:", train_index, "TEST:", test_index)
        # print('% train ', len(train_index) / (len(train_index) + len(test_index)))
        # print('Y=1|train ', np.sum(y[test_index]) / len(test_index))
        # print('Y=1|test ', np.sum(y[train_index]) / len(train_index))
        if split_i == split:
            print(' Split : ',split_i, ' is loaded...')
            dataset_list = ['train' if ix in train_index else 'test' for ix in range(len(pd_data))]
            pd_data['dataset'] = dataset_list
            print(np.unique(np.array(dataset_list)[np.array(train_index).astype('int')]),
                  np.unique(np.array(dataset_list)[np.array(test_index).astype('int')]))  # sanity check
        # print()
        split_i += 1

    pd_train = pd_data.loc[pd_data.dataset == 'train']
    pd_test = pd_data.loc[pd_data.dataset == 'test']

    ### Normalize
    if norm_std:
        ignore_tags = ['utility', 'groups']
        pd_full = pd.concat([pd_train, pd_test])
        mean_std_dict = compute_mean_std_tables(pd_full, cov_tags=cov_tags, save_dir=dirdata_lawschool)
        for c in mean_std_dict.keys():
            if c in ignore_tags:
                print(c)
                continue
            mean, std = mean_std_dict[c]
            if std > 0:
                pd_train[c] = (pd_train[c].values - mean) / std
                pd_test[c] = (pd_test[c].values - mean) / std
            else:
                print('problematic ', c, '; (mean,std) ', mean, std)

    check = all(item in pd_train['sample_index'].values for item in pd_test['sample_index'].values)
    print('------- Law school admission Dataset processing ---------- ')
    print('utility : ', utility, '; stratification_tags : ', strat_list, '; standarization : ',
          norm_std, '; len(cov_tags) : ', len(cov_tags))
    print('cov_tags : ', cov_tags)
    print('ntrain : ', len(pd_train), ' ; ntest : ', len(pd_test))
    print('check: overlap train and test ? ', check)
    print('')

    return pd_train, pd_test, cov_tags

##### Synthetic 1D DATASET #####
dirdata_synthetic = '/data/MLTdata/synthetic/dataset_processed/'

def synthetic_pandas( datatype='1d_2classes', utility='y', norm_std=True, seed=42,split = 1,n_splits=5):


    if datatype == '1d_2classes':
        pd_data = pd.read_csv(os.path.join(dirdata_synthetic, 'bpf_dataset_1covariate_2classes.csv'))
        cov_tags = ['x']

    ## set utility variables and reincorporate samples index
    pd_data['utility'] = pd_data[utility].values
    pd_data['sample_index'] = np.arange(len(pd_data))

    ### Train - Test split ###
    strat_list = ['utility'] #stratification tags for balance splitting
    values_strat =  pd_data['utility'].values
    pd_data['strat'] = values_strat

    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    X = pd_data.values
    y = np.array(values_strat)
    split_i = 1
    if split < split_i:
        print('SPLIT < 1 setting to : 1')
        split = 1
    if split > n_splits:
        print('SPLIT > n_splits setting to n_splits ')
        split = n_splits
    for train_index, test_index in skf.split(X, y):
        # print('*** split: ', split_i)
        # print("TRAIN:", train_index, "TEST:", test_index)
        # print('% train ', len(train_index) / (len(train_index) + len(test_index)))
        if split_i == split:
            print(' Split : ',split_i, ' is loaded...')
            dataset_list = ['train' if ix in train_index else 'test' for ix in range(len(pd_data))]
            pd_data['dataset'] = dataset_list
            print(np.unique(np.array(dataset_list)[np.array(train_index).astype('int')]),
                  np.unique(np.array(dataset_list)[np.array(test_index).astype('int')]))  # sanity check
        # print()
        split_i += 1

    pd_train = pd_data.loc[pd_data.dataset == 'train']
    pd_test = pd_data.loc[pd_data.dataset == 'test']

    ### Normalize
    if norm_std:
        ignore_tags = ['utility', 'groups']
        pd_full = pd.concat([pd_train, pd_test])
        mean_std_dict = compute_mean_std_tables(pd_full, cov_tags=cov_tags, save_dir=None)
        for c in mean_std_dict.keys():
            if c in ignore_tags:
                print(c)
                continue
            mean, std = mean_std_dict[c]
            if std > 0:
                pd_train[c] = (pd_train[c].values - mean) / std
                pd_test[c] = (pd_test[c].values - mean) / std
            else:
                print('problematic ', c, '; (mean,std) ', mean, std)

    check = all(item in pd_train['sample_index'].values for item in pd_test['sample_index'].values)
    print('------- Synthetic Dataset processing ---------- ')
    print('type : ', type)
    print('utility : ', utility, '; stratification_tags : ', strat_list, '; standarization : ',
          norm_std, '; len(cov_tags) : ',len(cov_tags))
    print('ntrain : ', len(pd_train), ' ; ntest : ', len(pd_test))
    print('check: overlap train and test ? ', check)
    print('')

    return pd_train, pd_test, cov_tags

