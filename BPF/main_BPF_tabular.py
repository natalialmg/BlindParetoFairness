import sys
import os
sys.path.append("../")
import argparse
import numpy as np

import pandas as pd
from distutils.util import strtobool
from general.utils import save_json,load_json

cparser = argparse.ArgumentParser()

## General specs
cparser.add_argument('--gpu', action='store', default=0, type=int,help='gpu')
cparser.add_argument('--model_name', action='store', default='BPF', type=str, help='model_name')
cparser.add_argument('--basedir', action='store', default='/data/natalia/models/', type=str,help='basedir for internal model save')
cparser.add_argument('--seed', action='store', default=42, type=int, help='randomizer seed')
cparser.add_argument('--train', action='store', default=True,type=lambda x: bool(strtobool(x)),help='boolean: train model?')

## Dataset
cparser.add_argument('--dataset', action='store', default='adult', type=str,help='dataset')
cparser.add_argument('--utility', action='store', default='', type=str,help='utility string tags')
cparser.add_argument('--norm_std', action='store', default=True,type=lambda x: bool(strtobool(x)),help='boolean: apply standarization to covariantes')
cparser.add_argument('--split', action='store', default=1, type=int,help='split for dataset partitions if applicable')
cparser.add_argument('--nsplits', action='store', default=5, type=int,help='n splits for dataset partitions if applicable')

## Dataloaders
cparser.add_argument('--batch', action='store', default=128, type=int, help='batch size')

##Games
cparser.add_argument('--loss', action='store', default='CE', type=str,help='loss CE, L2, L1')
cparser.add_argument('--games', action='store', default=200, type=int, help='number of games')
cparser.add_argument('--games_warmup', action='store', default=80, type=int, help='minimum number of games')
cparser.add_argument('--patience', action='store', default=15, type=int, help='no improvement worst loss patience')
cparser.add_argument('--valstop', action='store', default=False,type=lambda x: bool(strtobool(x)),help='boolean: validation stopper')

## Model Learner
cparser.add_argument('--hlayers', action='store', default='512x1', type=str,help='hidden layers widthxdepth e.g.: 128x4')
cparser.add_argument('--epoch_learner', action='store', default=1, type=int, help='SGD epochs learner does in each game')
cparser.add_argument('--epoch_learner_warmup', action='store', default=1, type=int, help='SGD epochs learner warmup training before first game') #usually in 1 means no warmup
cparser.add_argument('--lr', action='store', default=1e-5, type=float, help='learners learning rate ')
cparser.add_argument('--optim', action='store', default='adam', type=str,help='Learners optimizer')
cparser.add_argument('--batchnorm', action='store', default=False,type=lambda x: bool(strtobool(x)),help='boolean: batchnorm (default false)')
cparser.add_argument('--regression', action='store', default=False,type=lambda x: bool(strtobool(x)),help='boolean: regression (default false)')
cparser.add_argument('--modality', action='store', default='impweight', type=str,help='modality type : impweight, sampler, groups')

## Regulator
cparser.add_argument('--dreg', action='store', default=0.05, type=float, help='max %delta cost improve regulator (trust region) ')
cparser.add_argument('--epoch_reg', action='store', default=500, type=int, help='regulator ascent steps iterations in each game') #set this high since the stopping criteria is when dcost > dreg
cparser.add_argument('--dwreg', action='store', default=0.05, type=float, help='max %delta weight improve regulator (trust region) ')
# projector
cparser.add_argument('--eta', action='store', default=5, type=float, help='regulator learning rate ')
cparser.add_argument('--rho', action='store', default=1, type=float, help=' mean(weights) >= rho  ')
cparser.add_argument('--epsilon', action='store', default=1e-2, type=float, help='weights > epsilon ')
cparser.add_argument('--lrdecay', action='store', default=0.75, type=float, help='learning rate decay')
cparser.add_argument('--upper', action='store', default=1, type=float, help='weights < upper ')

#regularization L2 on adam
cparser.add_argument('--optim_regw', action='store', default=0, type=float, help='regularization weight')

cparser = cparser.parse_args()

datasets_included = ['adult','compas', 'lawschool','lawschool_nofaminc','synth1d2c','synth2d4c' ]

if __name__== '__main__':

    ### Data Setting

    from dataloaders.datasets import UCIadult_pandas, Compas_pandas, lawschool_pandas,synthetic_pandas
    from dataloaders.data_preprocessing import balanced_split

    if cparser.dataset not in datasets_included:
        print('WARNING:: Dataset ', cparser.dataset, ' is not supported !!!!!!! ')

    if cparser.dataset == 'adult':
        if cparser.utility not in ['income']:
            cparser.utility = 'income'
        pd_train, pd_test, cov_tags = UCIadult_pandas(utility=cparser.utility,
                                                      norm_std=cparser.norm_std,
                                                      seed=cparser.seed,
                                                      split=cparser.split,
                                                      n_splits=cparser.nsplits)

    if cparser.dataset == 'compas':
        cparser.utility = 'two_year_recid'
        pd_train, pd_test, cov_tags = Compas_pandas(utility=cparser.utility,
                                                    norm_std=cparser.norm_std,
                                                    seed=cparser.seed,
                                                    split=cparser.split,
                                                    n_splits=cparser.nsplits)

    if cparser.dataset == 'lawschool' :
        if cparser.utility not in ['pass_bar','zfygpa']:
            if cparser.regression:
                cparser.utility = 'zfygpa'
            else:
                cparser.utility = 'pass_bar'
        pd_train, pd_test, cov_tags = lawschool_pandas(utility=cparser.utility,
                                                    norm_std=cparser.norm_std,
                                                       seed=cparser.seed,
                                                       split=cparser.split,
                                                       n_splits=cparser.nsplits)

    if cparser.dataset == 'lawschool_nofaminc' :
        if cparser.utility not in ['pass_bar','zfygpa']:
            if cparser.regression:
                cparser.utility = 'zfygpa'
            else:
                cparser.utility = 'pass_bar'
        pd_train, pd_test, cov_tags = lawschool_pandas(utility=cparser.utility,
                                                       groups_list=['sex', 'race_bin', 'fam_inc'],
                                                       norm_std=cparser.norm_std,
                                                       seed=cparser.seed,
                                                       split=cparser.split,
                                                       n_splits=cparser.nsplits)



    if cparser.dataset == 'synth1d2c' :
        cparser.utility = 'y'
        pd_train, pd_test, cov_tags = synthetic_pandas(datatype='1d_2classes', utility=cparser.utility,
                                                        norm_std=cparser.norm_std,
                                                        seed=cparser.seed,
                                                        split=cparser.split,
                                                        n_splits=cparser.nsplits)

    if cparser.dataset == 'synth2d4c' :
        cparser.utility = 'y'
        pd_train, pd_test, cov_tags = synthetic_pandas(datatype='2d_4classes', utility=cparser.utility,
                                                        norm_std=cparser.norm_std,
                                                        seed=cparser.seed,
                                                        split=cparser.split,
                                                        n_splits=cparser.nsplits)



    ## Split train, into train and val balancing for the stratification tag (dataset dependent)
    tag = 'strat'
    ptrain = 0.7
    pval = 0.3

    pd_split_dic = balanced_split(pd_train, tag, p_vector=[ptrain, pval],
                                  seed=cparser.seed)
    pd_train = pd_split_dic[0]
    pd_val = pd_split_dic[1]

    print('seed : ',cparser.seed, '; split : ',cparser.split, ' of : ',cparser.nsplits)

    print('train : ', len(pd_train), ' samples')
    print(pd_train.groupby([tag])['utility'].count() / len(pd_train))
    print()
    print('val : ', len(pd_val), ' samples')
    print(pd_val.groupby([tag])['utility'].count() / len(pd_val))
    print()
    print('test : ', len(pd_test), ' samples')
    print(pd_test.groupby([tag])['utility'].count() / len(pd_test))
    print()

    print('group, E[utility|train] ,  E[utility|val] , E[utility|test] ')

    print(np.mean(pd_train['utility']),np.mean(pd_val['utility']),np.mean(pd_test['utility']))
    print()
    print()
    print()

    if cparser.regression:
        n_utility = 1
    else:
        n_utility = len(pd_train['utility'].unique())

    ## Model setting ##
    if cparser.hlayers is not '':
        if 'x' in cparser.hlayers:
            hlayers = cparser.hlayers.split('x')
            hlayers = tuple([int(hlayers[0]) for i in np.arange(int(hlayers[1]))])
        else:
            if '_' in cparser.hlayers:
                hlayers = cparser.hlayers.split('_')
                hlayers = tuple([int(i) for i in hlayers])
            else:
                hlayers = tuple([int(cparser.hlayers)])

    else:
        hlayers = ()

    from BPF.BPF_classes import BPF_config
    config = BPF_config(seed = cparser.seed,
                        GPU_ID=cparser.gpu,
                        n_utility=n_utility,
                        basedir=cparser.basedir,
                        model_name=cparser.model_name,
                        n_features=len(cov_tags),
                        hidden_layers=hlayers,
                        BATCH_SIZE=cparser.batch,
                        EPOCHS_LEARNER=cparser.epoch_learner,
                        EPOCHS_LEARNER_WARMUP=cparser.epoch_learner_warmup,
                        LEARNING_RATE=cparser.lr,
                        optimizer=cparser.optim,
                        type_loss=cparser.loss,
                        GAMES = cparser.games,
                        GAMES_WARMUP = cparser.games_warmup,
                        EPOCHS_REGULATOR=cparser.epoch_reg,
                        delta_improve_regulator=cparser.dreg,
                        delta_weight_change_regulator = cparser.dwreg,
                        eta = cparser.eta,
                        epsilon=cparser.epsilon,
                        rho=cparser.rho,
                        batchnorm = cparser.batchnorm,
                        lrdecay = cparser.lrdecay,
                        val_stopper= cparser.valstop,
                        modality=cparser.modality,
                        regression=cparser.regression,
                        patience = cparser.patience,
                        optim_weight_decay = cparser.optim_regw)

    ######################### Dataloader functional  ##############################
    from dataloaders.dataloaders import get_dataloaders_tabular
    def dataloader_functional(pd_train, sampler_on=False,
                              sampler_tag=config.sampler_tag, shuffle=True, weights_tag=None):
        return get_dataloaders_tabular(pd_train, utility_tag='utility', cov_tags=cov_tags,
                                       sampler_tag=sampler_tag,
                                       weights_tag=weights_tag,
                                       sampler_on=sampler_on, num_workers=config.n_workers,
                                       batch_size=config.BATCH_SIZE, regression=config.regression, shuffle=shuffle)


    ######################## BPF Model ############################################
    from BPF.BPF_classes import BPF_model
    model = BPF_model(config=config)

    if cparser.train:
        history = model.train(pd_train, pd_val, dataloader_functional)
    else:
        history = load_json(model.config.basedir + model.config.model_name + '/history.json')


    ############################## Evaluation ##############################
    dataset_tag = ['train', 'val', 'test']
    list_pd = [pd_train, pd_val, pd_test]

    pd_summary_list = ['/summary_performance.csv', '/summary_performance_train.csv']
    model_list = [model.config.best_model, model.config.best_model_train]

    for ix_model in range(len(model_list)):

        if os.path.exists(model.config.basedir + model.config.model_name + '/' + model_list[ix_model]):
            print('Loading model for evaluation :: ', model_list[ix_model])
            model.load_network(load_file=model.config.basedir + model.config.model_name + '/' + model_list[ix_model])
        else:
            continue

        ix = 0
        for pd_data in list_pd:
            eval_dataloader = dataloader_functional(pd_data, sampler_on=False, shuffle=False, weights_tag=None)
            output = model.eval(eval_dataloader)

            # Save evaluation
            y_pred_tags = ['utility_pest_' + str(_) for _ in range(config.n_utility)]
            pd_results_ix = pd.DataFrame(data=np.array(output['utility_pred']).transpose(), columns=y_pred_tags)
            pd_results_ix['utility_gt'] = output['utility_gt']
            pd_results_ix['sample_index'] = pd_data['sample_index'].values
            pd_results_ix['dataset'] = dataset_tag[ix]
            if ix == 0:
                pd_results = pd_results_ix.copy()
            else:
                pd_results = pd.concat([pd_results, pd_results_ix])
            ix += 1

        from general.evaluation import get_output_list, get_cross_entropy, get_brier_score, get_error, get_soft_error
        y_pred, y_gt = get_output_list(pd_results, y_pred_tags, y_gt_tag='utility_gt')
        metrics_fn = [get_cross_entropy, get_brier_score, get_error, get_soft_error]
        metrics_tags = ['ce', 'bs', 'err', 'softerr']

        for ix in np.arange(len(metrics_tags)):
            metric_results = metrics_fn[ix](y_gt, y_pred)
            pd_results[metrics_tags[ix]] = metric_results

        pd_results.to_csv(config.basedir + config.model_name + '/pd_eval.csv', index=0)
        print('Evaluation csv saved on : ', config.basedir + config.model_name + '/pd_eval.csv')

        ######################### Summary results  ##############################

        # dataframe with columns: dataset, rho_eval, metric, worst, best, avg
        # rho_eval = np.linspace(0, 1, 11)[1:-1]
        rho_eval = np.linspace(0, 1, 21)[1:-1]
        row = []
        for dataset in dataset_tag:
            pd_filter = pd_results.loc[pd_results.dataset == dataset]
            for metric in metrics_tags:
                mvalues = pd_filter[metric].values
                mvalues = np.sort(mvalues)[::-1]
                for rho in rho_eval:
                    nworst = int(np.floor(mvalues.shape[0] * rho))
                    worst_group = np.mean(mvalues[0:nworst])
                    best_group = np.mean(mvalues[nworst:])
                    row.append([dataset, metric, rho, worst_group,
                                best_group, np.mean(mvalues)])

        pd_summary = pd.DataFrame(data=row, columns=['dataset', 'metric', 'rho_eval', 'worst', 'best', 'avg'])

        # include split,epsilon_model, rho_model
        pd_summary['split'] = cparser.split
        pd_summary['epsilon_model'] = config.epsilon
        pd_summary['rho_model'] = config.rho

        pd_summary.to_csv(config.basedir + config.model_name + pd_summary_list[ix_model], index=0)
        print('Evaluation csv saved on : ', config.basedir + config.model_name + pd_summary_list[ix_model])
