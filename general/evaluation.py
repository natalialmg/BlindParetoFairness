
import numpy as np
from .utils import TravellingMean, to_np
import torch.nn as nn

def epoch_persample_evaluation(dataloader, classifier_network, criterion, DEVICE,metrics_dic = None,regression = False):

    # dataloader
    # classifier_network
    # criterion: loss function

    classifier_network.eval()

    output = {}
    output['loss'] = []
    output['utility_pred'] = []
    output['utility_gt'] = []

    if metrics_dic is not None:
        for key in metrics_dic.keys():
            output[key] = TravellingMean()

    start = True
    for i_batch, sample_batch in enumerate(dataloader):
        x, utility = sample_batch
        x = x.to(DEVICE)
        utility = utility.to(DEVICE)  # batch x nutility

        # get output and loss
        logits = classifier_network(x)

        ######## Loss #########
        loss = criterion(logits, utility) #size batch
        output['loss'].extend(list(to_np(loss).flatten()))

        #######  Metrics #######

        if metrics_dic is not None:
            for key in metrics_dic.keys():
                metric = metrics_dic[key](logits,utility)
                output[key].extend(list(metric))

        #######  Save predictions #######
        if not regression:
            prediction = to_np(nn.Softmax(dim=-1)(logits))
        else:
            prediction = to_np(prediction)

        if start:
            output['utility_pred'] = [[] for _ in range(utility.shape[1])]
            output['utility_gt'] = []
            start = False

        for _ in range(utility.shape[1]):
            output['utility_pred'][_].extend(list(prediction[:, _]))
        if not regression:
            output['utility_gt'].extend(list(to_np(utility).argmax(-1)))
        else:
            output['utility_gt'].extend(list(to_np(utility).flatten()))

    return output


def get_output_list(pd_data,y_pred_tags, y_gt_tag = 'utility_gt',groups_gt_tag = None, regression = False):
    y_pred = []
    y_gt = []
    groups_gt = []
    if groups_gt_tag is not None:
        groups_gt = pd_data[groups_gt_tag].values

    if not regression:
        y_gt_aux = pd_data[y_gt_tag].values
    for i in range(len(y_pred_tags)):
        y_pred.append(pd_data[y_pred_tags[i]].values)
        if not regression:
            y_gt.append((y_gt_aux == i).astype('int'))

    if regression:
        y_gt = pd_data[y_gt_tag].values
        y_gt = y_gt[:,np.newaxis]
    else:
        y_gt = np.array(y_gt).transpose()

    y_pred = np.array(y_pred).transpose()

    if groups_gt_tag is not None:
        return y_pred,y_gt,groups_gt
    else:
        return y_pred, y_gt

def get_cross_entropy(y_gt,y_est,sample_weight = None):
    # y_gt : number of samples x number of classes
    # y_est: number of samples x number of classes
    # weigths : number of samples
    if sample_weight is None:
        return (-1*np.sum(y_gt*np.log(np.maximum(y_est,1e-20)),axis = 1))
    else:
        return np.sum(sample_weight*(-1*np.sum(y_gt*np.log(np.maximum(y_est,1e-20)),axis = 1)))/np.sum(sample_weight)

def get_brier_score(y_gt,y_est,sample_weight = None):
    # y_gt : number of samples x number of classes
    # y_est: number of samples x number of classes
    # weigths : number of samples

    if sample_weight is None:
        return (np.sum((y_gt - y_est)**2,axis = 1))
    else:
        return np.sum(sample_weight*np.sum((y_gt - y_est)**2,axis = 1))/np.sum(sample_weight)

def get_soft_error(y_gt,y_est,sample_weight = None):
    # y_gt : number of samples x number of classes
    # y_est: number of samples x number of classes
    # weigths : number of samples
    ixmax_gt = np.argmax(y_gt,axis=1)
    # ixmax_est = np.argmax(y_est,axis=1)
    if sample_weight is None:
        return 1-y_est[np.arange(ixmax_gt.shape[0]),ixmax_gt]
        # return ixmax_gt
    else:
        return np.sum(sample_weight*(1-y_est[np.arange(ixmax_gt.shape[0]),ixmax_gt]))/np.sum(sample_weight)

def get_error(y_gt,y_est,sample_weight = None):
    # y_gt : number of samples x number of classes
    # y_est: number of samples x number of classes
    # weigths : number of samples
    ixmax_gt = np.argmax(y_gt,axis=1)
    ixmax_est = np.argmax(y_est,axis=1)
    if sample_weight is None:
        return (ixmax_gt != ixmax_est).astype('int')
    else:
        return np.sum(sample_weight*(ixmax_gt != ixmax_est).astype('int'))/np.sum(sample_weight)


import pandas as pd
def get_best_model(pd_summary, model_tag='rho_model', dataset_choice=['train', 'val'], precision=3):
    matching_values = []  # split, metric, rho_eval, best model for rho eval
    for split in pd_summary.split.unique():
        filter_rows = np.ones([len(pd_summary)]) != 1
        for dataset in dataset_choice:
            filter_rows = (filter_rows) | (pd_summary.dataset == dataset)
        filter_rows = filter_rows & (pd_summary.split == split)

        for metric in pd_summary.metric.unique():
            for rho_eval in pd_summary.rho_eval.unique():
                pd_aux = pd_summary.loc[filter_rows & (pd_summary.metric == metric) & (pd_summary.rho_eval == rho_eval)]

                worst = np.array(np.round(pd_aux.groupby(model_tag)['worst'].mean(), precision))
                min_worst = np.min(worst)
                min_worst_mask = (worst == min_worst)

                best = np.array(np.round(pd_aux.groupby(model_tag)['best'].mean(), precision)) * min_worst_mask
                best += (np.max(best) + 10) * (1 - min_worst_mask)  # mask out non min worst values
                arg_choice = int(np.argmin(best))

                best_model_tag = np.array(pd_aux.groupby(model_tag)[model_tag].mean())
                best_model_tag = best_model_tag[arg_choice]
                matching_values.append([split, metric, rho_eval, best_model_tag])

    matching_values = pd.DataFrame(data=matching_values, columns=['split', 'metric', 'rho_eval', model_tag])

    ## Generate pandas with Best model per rho evaluation
    pd_out = None
    for split in matching_values.split.unique():
        for metric in matching_values.metric.unique():
            for rho_eval in matching_values.rho_eval.unique():
                matching_values_aux = matching_values.loc[
                    (matching_values.split == split) & (matching_values.metric == metric) &
                    (matching_values.rho_eval == rho_eval)]

                best_model_tag = matching_values_aux[model_tag].values[0]
                pd_out_aux = pd_summary.loc[(pd_summary.split == split) & (pd_summary.metric == metric) &
                                            (pd_summary.rho_eval == rho_eval) & (
                                            pd_summary[model_tag] == best_model_tag)]

                if pd_out is None:
                    pd_out = pd_out_aux.copy()
                else:
                    pd_out = pd.concat([pd_out, pd_out_aux])

    return pd_out