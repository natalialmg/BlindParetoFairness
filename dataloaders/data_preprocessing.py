
import numpy as np
import pandas as pd
import json,os

def extract_sensitive_columns(s_columns,pd_table,s_inc_col = False):
    #s_inc_col : tag s included in column (less restrictive than being == column tag)

    ## drop columns ##
    drop_columns = []
    s_dic = {}
    for s in s_columns:
        s_dic[s] = pd.DataFrame()

    for col in pd_table.columns:
        for s in s_columns:
            if ((s in col)&(s_inc_col)) | (s == col):
                s_dic[s][col] = pd_table[col].values
                drop_columns.append(col)
    pd_out = pd_table.drop(columns=drop_columns)
    return pd_out,s_dic

def group_concatenation(groups_dic, groups_list, tag2int={}, int2tag=[]):
    cont = len(int2tag)
    # print(cont)
    nelem = len(groups_dic[groups_list[0]])
    values = []
    for i in np.arange(nelem):

        g_list_i = []
        g_str = ''
        for g in groups_list:
            g_val = groups_dic[g].values[i][0]
            g_list_i.append(g_val)
            g_str = g_str + str(g_val) + ','

        if g_str not in tag2int.keys():
            tag2int[g_str] = cont
            g_list_i.append(g_str)
            int2tag.append(g_list_i)
            cont += 1
        values.append(tag2int[g_str])

    return values, int2tag, tag2int

def compute_mean_std_tables(pd_table,cov_tags = [],save_dir = None):
    temp_dict = pd_table.describe().to_dict()
    mean_std_dict = {}
    for key, value in temp_dict.items():
        if key in cov_tags:
            mean_std_dict[key] = [value['mean'],value['std']]
    if save_dir is not None:
        output_file_path = os.path.join(save_dir,'mean_std.json')
        with open(output_file_path, mode="w") as output_file:
            output_file.write(json.dumps(mean_std_dict))
            output_file.close()
    return mean_std_dict

def balanced_split(pd_data, tag, p_vector=[0.5, 0.5], seed=0):
    if seed > 0:
        np.random.seed(seed)


    index_splits = [[] for i in range(len(p_vector))]

    pd_data = pd_data.copy()
    pd_data['id'] = list(np.arange(len(pd_data))) #generate column with row id's (indexes)
    for u in pd_data[tag].unique():
        pd_filter = pd_data.loc[pd_data[tag] == u]
        index_values = np.array(pd_filter['id'].values)+0 # gen indexes
        np.random.shuffle(index_values)

        nitems = len(index_values)
        ix = 0
        for s in range(len(p_vector)):
            if s == (len(p_vector)-1):
                index_splits[s].extend(index_values[ix:])
            else:
                index_splits[s].extend(index_values[ix:ix + int(nitems * p_vector[s])])
            ix += int(nitems * p_vector[s])

    pd_split_dic = {}
    pd_data.drop(columns=['id'])
    for s in range(len(p_vector)):
        # print(len(index_splits[s]))
        pd_split_dic[s] = pd_data.iloc[index_splits[s]].copy() #get rows per group


    return pd_split_dic



