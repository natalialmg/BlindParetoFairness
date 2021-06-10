import numpy as np
import torch
import sys
sys.path.append("../")
from general.utils import TravellingMean, to_np, model_params_load,model_params_save
from general.evaluation import epoch_persample_evaluation


def BPF_trainer(dataloader_functional, pd_train, pd_val,
                   optimizer, classifier_network, criterion, config, BPF_projector,
                    metrics_dic = None, reg_dic=None, reg_weights=None, precision = 3,
                             val_stopper=True,warmup=80):

    # from config uses : config.EPOCHS, config.lrdecay, config.DEVICE, config.best_adaptive_network_path, config.patience
    # learning_rate_learner = np.zeros([config.GAMES*config.EPOCHS_LEARNER])
    # learning_rate_regulator = np.zeros([config.GAMES*config.EPOCHS_LEARNER])

    # history_tags = ['loss','criterion']
    # history_tags = ['loss','loss_mbatch']
    history = {}
    history['game'] = []
    history['epoch'] = []
    history['learning_rate_learner'] = []
    history['learning_rate_regulator'] = []

    #loss mbatch
    history['loss_a1_mbatch_train'] = []
    history['loss_a0_mbatch_train'] = []
    history['criterion_mbatch_train'] = []

    #current game importance weight loss (after learner update)
    history['delta_weights_train'] = []
    history['loss_iw_a1_train'] = []
    history['loss_iw_a0_train'] = []

    if reg_dic is not None:
        for reg in reg_dic.keys():
            history[reg + '_mbatch_train'] = []


    if metrics_dic is not None:
        for metric in metrics_dic.keys():
            history[metric + '_mbatch_train'] = []

            history[metric + '_a1_train'] = []
            history[metric + '_a0_train'] = []
            history[metric + '_average_train'] = []

            history[metric + '_a1_val'] = []
            history[metric + '_a0_val'] = []
            history[metric + '_average_val'] = []

    #losses general
    history['loss_a1_train'] = []
    history['loss_a0_train'] = []
    history['loss_average_train'] = []

    history['loss_a1_val'] = []
    history['loss_a0_val'] = []
    history['loss_average_val'] = []


    game = 0
    best_game = {'train': 0, 'val': 0}
    best_loss_worst = {'train': np.inf, 'val': np.inf}
    best_average_loss = {'train': np.inf, 'val': np.inf}
    patience = 0
    # best_worst_loss_gap = np.inf
    sample_weights = np.ones([len(pd_train)])*config.rho # initialize sample weights with prior


    # Evaluation dataloaders
    val_eval_dataloader = dataloader_functional(pd_val, sampler_on=False, shuffle=False, weights_tag=None)
    train_eval_dataloader = dataloader_functional(pd_train, sampler_on=False, shuffle=False, weights_tag=None)

    if config.modality == 'impweight':  # this is default modality
        print('- mode : impweight')
        pd_train[config.weights_tag] = sample_weights / np.mean(sample_weights)
        train_dataloader = dataloader_functional(pd_train, weights_tag=config.weights_tag,
                                                 sampler_on=False, sampler_tag=config.sampler_tag)

    # LEARNING_RATE = optimizer.param_groups[0]['lr']
    # lr_cooldown = 0

    while (game < config.GAMES):

        print('----- GAME : ',game, '/', config.GAMES ,'------')

        ######################### Learner Plays #########################

        print('*Learner*')
        print('mean/min/max sample weights : ,',np.round(np.mean(sample_weights),precision),
              np.round(np.min(sample_weights),precision),np.round(np.max(sample_weights),precision))

        if config.modality == 'impweight': #this is default modality
            W_torch = torch.Tensor(np.vstack(sample_weights / np.mean(sample_weights)).astype('float32'))
            train_dataloader.dataset.update_attributes(W_torch=W_torch)

        if config.modality == 'sampler':
            print('-  mode : sampler')
            pd_train[config.weights_tag] = 1.0  # importance weight set to 1, sampler is sampling according to sample_weights, requires redefine dataloader in each iteration
            pd_train[config.sampler_tag] = sample_weights/np.sum(sample_weights)
            train_dataloader = dataloader_functional(pd_train, weights_tag=config.weights_tag,
                                                     sampler_on=True, sampler_tag=config.sampler_tag)

        if (game == 0) & (config.EPOCHS_LEARNER_WARMUP>config.EPOCHS_LEARNER):
            EPOCHS_LEARNER = config.EPOCHS_LEARNER_WARMUP
        else:
            EPOCHS_LEARNER = config.EPOCHS_LEARNER

        for epoch in range(config.EPOCHS_LEARNER):

            history['learning_rate_learner'].append(optimizer.param_groups[0]['lr'] + 0)

            output = BPF_epoch_learner(train_dataloader, classifier_network, criterion, optimizer, config.DEVICE, metrics_dic=metrics_dic,
                              reg_dic=reg_dic, reg_weights=reg_weights, train_type=True)

            history['loss_a1_mbatch_train'].append(output['loss'])
            history['loss_a0_mbatch_train'].append(output['comp_loss'])
            history['criterion_mbatch_train'].append(output['criterion'])

            if reg_dic is not None:
                for reg in reg_dic.keys():
                    history[reg+'_mbatch_train'].extend(output[reg])

            if metrics_dic is not None:
                for metric in metrics_dic.keys():
                    history[metric + '_mbatch_train'].extend(output[metric])
        #####################################################


        ########## Evaluation on worst case partition (train and validation) ############
        output_train = epoch_persample_evaluation(train_eval_dataloader, classifier_network, criterion, config.DEVICE, metrics_dic=metrics_dic)
        output_val = epoch_persample_evaluation(val_eval_dataloader, classifier_network, criterion, config.DEVICE,metrics_dic=metrics_dic)

        weight_worst_train = BPF_projector.get_worst_weights(np.array(output_train['loss']))
        weight_worst_val  = BPF_projector.get_worst_weights(np.array(output_val['loss']))

        print('Sanity check: ')
        print('mean/min/max worst_weights_train : ,',np.round(np.mean(weight_worst_train),precision),
              np.round(np.min(weight_worst_train),precision),np.round(np.max(weight_worst_train),precision))
        print('mean/min/max worst_weights_val : ,', np.round(np.mean(weight_worst_val), precision),
              np.round(np.min(weight_worst_val), precision), np.round(np.max(weight_worst_val), precision))
        print()

        #train worst partition losses
        den_a1_train = np.mean(weight_worst_train) if np.mean(weight_worst_train)>0 else 1e-20 #denominator
        den_a0_train = (BPF_projector.upper-den_a1_train) if BPF_projector.upper-den_a1_train>0 else 1e-20

        loss_a1_train = np.round(np.mean(weight_worst_train*np.array(output_train['loss']))/den_a1_train,precision)
        loss_a0_train = np.round(np.mean(np.abs(BPF_projector.upper-weight_worst_train) * np.array(output_train['loss']))/den_a0_train,precision) #division by 0
        average_loss_train = np.round(np.mean(np.array(output_train['loss'])), precision)

        # val worst partition losses
        den_a1_val = np.mean(weight_worst_val) if np.mean(weight_worst_val) > 0 else 1e-20 #denominator
        den_a0_val = (BPF_projector.upper - den_a1_val) if BPF_projector.upper - den_a1_val > 0 else 1e-20

        loss_a1_val = np.round(np.mean(weight_worst_val * np.array(output_val['loss']))/den_a1_val,precision)
        loss_a0_val = np.round(np.mean(np.abs(BPF_projector.upper - weight_worst_val) * np.array(output_val['loss']))/den_a0_val,precision)
        average_loss_val = np.round(np.mean(np.array(output_val['loss'])),precision)

        ###############  Regulator Plays (new weight update) ###############
        print('*Regulator*')

        weight_list, _, _, flag_out = BPF_projector.update_weights(np.array(output_train['loss']), sample_weights,
                                                                   iterations=config.EPOCHS_REGULATOR)
        sample_weights = weight_list[-1]
        loss_iw_a1_train = np.round(
            np.mean(sample_weights * np.array(output_train['loss'])) / np.maximum(np.mean(sample_weights), 1e-20),
            precision)
        loss_iw_a0_train = np.round(
            np.mean(np.abs(BPF_projector.upper - sample_weights) * np.array(output_train['loss'])) / np.maximum(
                np.mean(np.abs(BPF_projector.upper - sample_weights)), 1e-20), precision)  # division by 0

        #### Saving worst case losses..
        history['delta_weights_train'].extend([np.sum(np.abs(weight_worst_train - sample_weights))/np.sum(sample_weights) for i in range(EPOCHS_LEARNER)])

        #first save train loss with the current sample weights!!!
        history['loss_iw_a1_train'].extend([loss_iw_a1_train for i in range(EPOCHS_LEARNER)])
        history['loss_iw_a0_train'].extend([loss_iw_a0_train for i in range(EPOCHS_LEARNER)])
        history['loss_a1_train'].extend([loss_a1_train for i in range(EPOCHS_LEARNER)])
        history['loss_a0_train'].extend([loss_a0_train for i in range(EPOCHS_LEARNER)])
        history['loss_average_train'].extend([average_loss_train for i in range(EPOCHS_LEARNER)])
        history['loss_a1_val'].extend([loss_a1_val for i in range(EPOCHS_LEARNER)])
        history['loss_a0_val'].extend([loss_a0_val for i in range(EPOCHS_LEARNER)])
        history['loss_average_val'].extend([average_loss_val for i in range(EPOCHS_LEARNER)])

        #metrics
        if metrics_dic is not None:
            for metric in metrics_dic.keys():
                worst_metric_a1_train = np.mean(weight_worst_train * np.array(output_train[metric]))/den_a1_train
                worst_metric_a0_train = np.mean((BPF_projector.upper - weight_worst_train) * np.array(output_train[metric]))/den_a0_train
                average_metric_train = np.mean(np.array(output_train[metric]))

                history[metric + '_a1_train'].extend([np.round(worst_metric_a1_train,precision) for i in range(EPOCHS_LEARNER)])
                history[metric + '_a0_train'].extend([np.round(worst_metric_a0_train,precision) for i in range(EPOCHS_LEARNER)])
                history[metric + '_average_train'].extend([np.round(average_metric_train, precision) for i in range(EPOCHS_LEARNER)])

                worst_metric_a1_val = np.mean(weight_worst_val * np.array(output_val[metric]))/den_a1_val
                worst_metric_a0_val = np.mean((BPF_projector.upper - weight_worst_val) * np.array(output_val[metric]))/den_a0_val
                average_metric_val = np.mean(np.array(output_val[metric]))

                history[metric + '_a1_val'].extend([np.round(worst_metric_a1_val, precision) for i in range(EPOCHS_LEARNER)])
                history[metric + '_a0_val'].extend([np.round(worst_metric_a0_val, precision) for i in range(EPOCHS_LEARNER)])
                history[metric + '_average_val'].extend([np.round(average_metric_val, precision) for i in range(EPOCHS_LEARNER)])

        history['game'].extend([game for _ in np.arange(EPOCHS_LEARNER)])
        history['epoch'].extend([_ for _ in np.arange(EPOCHS_LEARNER)])
        history['learning_rate_regulator'].extend([BPF_projector.eta for _ in np.arange(EPOCHS_LEARNER)])

        print('-Learner summary ')

        # # save last model
        # model_params_save(config.basedir + config.model_name + '/' + config.last_model, classifier_network, optimizer)
        # print('-----------')
        # print()


        ########## saving models ############
        patience += 1

        ## Val selection
        primary_loss = np.maximum(loss_a1_val, loss_a0_val)
        secondary_loss = np.minimum(loss_a1_val, loss_a0_val)
        if (primary_loss < best_loss_worst['val']) | ((primary_loss == best_loss_worst['val'])&(secondary_loss<best_average_loss['val'])) | (game == 0):
            print('saving best model val loss :', primary_loss, ' to :',config.best_model)
            model_params_save(config.basedir + config.model_name + '/' + config.best_model, classifier_network,optimizer)
            best_game['val'] = game + 0
            best_loss_worst['val'] = primary_loss + 0
            best_average_loss['val'] = secondary_loss + 0

            # save weights of best model
            history['weight_worst_val'] = weight_worst_val
            history['sample_val_loss'] = output_val['loss']

            if val_stopper:
                patience = 0

        ## Train selection
        primary_loss = np.maximum(loss_a1_train,loss_a0_train)
        secondary_loss = np.minimum(loss_a1_train,loss_a0_train)
        if (primary_loss < best_loss_worst['train']) | ((primary_loss == best_loss_worst['train'])&(secondary_loss<best_average_loss['train'])) | (game == 0):
            print('saving best model train loss :',primary_loss, ' to :',config.best_model_train)
            model_params_save(config.basedir + config.model_name + '/' + config.best_model_train, classifier_network,optimizer)
            best_game['train'] = game + 0
            best_loss_worst['train'] = primary_loss + 0
            best_average_loss['train'] = secondary_loss + 0

            # save weights of best model
            history['weight_worst_train'] = weight_worst_train
            history['sample_train_loss'] = output_train['loss']

            if not val_stopper:
                patience = 0

        # Best model, stopping criteria, and print
        # if val_stopper:
        #     primary_loss = np.maximum(loss_a1_val,loss_a0_val)
        #     secondary_loss = np.minimum(loss_a1_val,loss_a0_val)
        # else:
        #     primary_loss = np.maximum(loss_a1_train,loss_a0_train)
        #     secondary_loss = np.minimum(loss_a1_train,loss_a0_train)

        # if (primary_loss < best_primary_loss) | ((primary_loss == best_primary_loss)&(secondary_loss<best_secondary_loss)) | (game == 0):
        #     print('saving best model...')
        #     model_params_save(config.basedir + config.model_name + '/' + config.best_model, classifier_network,optimizer)
        #     best_game = game + 0
        #     best_primary_loss = primary_loss + 0
        #     best_secondary_loss = secondary_loss + 0
        #     patience = 0
        #
        #     #save weights of best model
        #     history['weight_worst_train'] = weight_worst_train
        #     history['weight_worst_val'] = weight_worst_val
        #     history['sample_weights'] = sample_weights
        #     history['sample_train_loss'] = output_train['loss']
        #     history['sample_val_loss'] = output_val['loss']
        #
        # else:
        #     patience += 1

        # Print -----
        string_print = 'Game: ' + str(game) + '; lr_L: ' + str(optimizer.param_groups[0]['lr']) + '; lr_R: ' + str(BPF_projector.eta) + '; weights_rho : ' +\
                       str(np.mean(sample_weights))
        string_print = string_print  + '| worst/best_iwloss_tr ' + str(np.round(loss_iw_a1_train, 3)) + ', ' + str(np.round(loss_iw_a0_train, 3)) +\
                       '| worst_loss_tr/val ' + str(np.round(loss_a1_train, 3)) + ', ' + str(np.round(loss_a1_val, 3)) + \
                       '| best_loss_tr/val ' + str(np.round(loss_a0_train, 3)) + ', ' + str(np.round(loss_a0_val, 3)) + \
                       '| avg_loss_tr/val ' + str(np.round(average_loss_train, 3)) + ', ' + str(np.round(average_loss_val, 3))

        string_print = string_print + '|primary current/best_tr/best_val ' + str(np.round(primary_loss, precision)) + ', ' +\
                       str(np.round(best_loss_worst['train'], precision)) + ', ' +  str(np.round(best_loss_worst['val'], precision)) +\
                      '|secondary current/best_tr/best_val ' + str(np.round(secondary_loss,precision)) +\
                       ', ' + str(np.round(best_average_loss['train'], precision)) + ', ' + str(np.round(best_average_loss['val'], precision)) +\
                      '| dif_weights : ' + str(np.round(np.sum(np.abs(weight_worst_train-sample_weights))/np.sum(sample_weights),precision))


        string_print = string_print + '|best game :'+str(best_game)+', patience : ',str(patience)

        print(string_print)
        print()
        print()

        # consider printing constraints ?,
        game += 1

        if (patience > config.patience) & (game > warmup):
            break

    # -------- END TRAINING --------#

    # load best network
    history['best_game'] = best_game
    print('Training Ended, loading best model : ', config.basedir + config.model_name + '/'+config.best_model)
    model_params_load(config.basedir + config.model_name + '/'+config.best_model, classifier_network, optimizer, config.DEVICE)

    return history


def BPF_epoch_learner(dataloader, classifier_network, criterion, optimizer, DEVICE, metrics_dic = None,
                           reg_dic=None, reg_weights=None, train_type=True):

    # dataloader
    # classifier_network
    # criterion: loss function
    # weights_group: numpy vector size groups and norm 1
    # reg_dic: dictionary with {name of regularization: regularization function}
    # reg_weights: dictionary with {name of regularization: weight}

    if train_type:
        classifier_network.train()
    else:
        classifier_network.eval()

    output = {}
    output['criterion'] = TravellingMean()
    output['loss'] = TravellingMean()
    output['comp_loss'] = TravellingMean() #complementarity loss

    if reg_dic is not None:
        for key in reg_dic.keys():
            output[key] = TravellingMean()

    if metrics_dic is not None:
        for key in metrics_dic.keys():
            output[key] = TravellingMean()
            output['comp_' + key] = TravellingMean()

    for i_batch, sample_batch in enumerate(dataloader):
        x, utility, weights = sample_batch
        x = x.to(DEVICE)
        utility = utility.to(DEVICE)  # batch x nutility
        # groups = groups.to(DEVICE) # batch x ngroups
        weights = weights.to(DEVICE) # batch x 1

        # zero the parameter gradients
        optimizer.zero_grad()

        # get output and loss
        logits = classifier_network(x)

        ######## Loss #########
        loss = criterion(logits, utility) #size batch
        weights_np = to_np(weights.flatten())
        comp_loss = np.mean(to_np(loss.flatten())*(1-weights_np))#/np.maximum(np.sum(1-weights_np),1e-20)  #todo:! we assume 1 is BPF_projection.upper.. to update later...

        # print(loss*weights)
        loss = torch.mean(loss.flatten()*weights.flatten())#/torch.sum(weights.flatten())
        output['loss'].update(np.array([to_np(loss)]),mass=to_np(weights).shape[0]) # update rolling loss
        output['comp_loss'].update(comp_loss, mass=to_np(weights).shape[0])  # update rolling comp_loss

        if reg_dic is not None:  #if we have regularizations
            for tag in reg_dic.keys(): #compute each regularization
                reg = reg_dic[tag]
                loss_r = reg.forward(classifier_network) #we assume they are a function only of the network parameters

                if reg_weights is None:
                    loss += loss_r
                    ## Update regularization
                    output[tag].update(np.array([to_np(loss_r)]))
                else:
                    loss += reg_weights[tag] * loss_r
                    ## Update regularization
                    output[tag].update(np.array([to_np(reg_weights[tag] * loss_r)]))

        # update rolling full loss (loss + reg)
        output['criterion'].update(np.array([to_np(loss)]),mass=to_np(weights).shape[0])

        if train_type:
            # backpropagation
            loss.backward()
            optimizer.step()

        #######  Metrics #######

        if metrics_dic is not None:
            for key in metrics_dic.keys():
                metric = metrics_dic[key](logits,utility)
                # Metrics mean
                metric = torch.mean(metric*weights)
                output[key].update(np.array([to_np(metric)]))

    for key in output.keys():
        output[key] = output[key].mean  # save only mean

    return output




# def BPF_trainer(dataloader_functional, pd_train, pd_val,
#                    optimizer, classifier_network, criterion, config, BPF_projector,
#                     metrics_dic = None, reg_dic=None, reg_weights=None, precision = 3,
#                              val_stopper=True,warmup=80):
#
#     # from config uses : config.EPOCHS, config.lrdecay, config.DEVICE, config.best_adaptive_network_path, config.patience
#     # learning_rate_learner = np.zeros([config.GAMES*config.EPOCHS_LEARNER])
#     # learning_rate_regulator = np.zeros([config.GAMES*config.EPOCHS_LEARNER])
#
#     # history_tags = ['loss','criterion']
#     # history_tags = ['loss','loss_mbatch']
#     history = {}
#     history['game'] = []
#     history['epoch'] = []
#     history['learning_rate_learner'] = []
#     history['learning_rate_regulator'] = []
#
#     #loss mbatch
#     history['loss_a1_mbatch_train'] = []
#     history['loss_a0_mbatch_train'] = []
#     history['criterion_mbatch_train'] = []
#
#     #current game importance weight loss (after learner update)
#     history['delta_weights_train'] = []
#     history['loss_iw_a1_train'] = []
#     history['loss_iw_a0_train'] = []
#
#     if reg_dic is not None:
#         for reg in reg_dic.keys():
#             history[reg + '_mbatch_train'] = []
#
#
#     if metrics_dic is not None:
#         for metric in metrics_dic.keys():
#             history[metric + '_mbatch_train'] = []
#
#             history[metric + '_a1_train'] = []
#             history[metric + '_a0_train'] = []
#             history[metric + '_average_train'] = []
#
#             history[metric + '_a1_val'] = []
#             history[metric + '_a0_val'] = []
#             history[metric + '_average_val'] = []
#
#     #losses general
#     history['loss_a1_train'] = []
#     history['loss_a0_train'] = []
#     history['loss_average_train'] = []
#
#     history['loss_a1_val'] = []
#     history['loss_a0_val'] = []
#     history['loss_average_val'] = []
#
#
#     game = 0
#     best_game = 0
#     best_primary_loss = np.inf #worst group loss
#     best_secondary_loss = np.inf #best group loss
#     # best_worst_loss_gap = np.inf
#     sample_weights = np.ones([len(pd_train)])*config.rho # initialize sample weights with prior
#
#
#
#     # patience = 0
#
#     # Evaluation dataloaders
#     val_eval_dataloader = dataloader_functional(pd_val, sampler_on=False, shuffle=False, weights_tag=None)
#     train_eval_dataloader = dataloader_functional(pd_train, sampler_on=False, shuffle=False, weights_tag=None)
#
#     # LEARNING_RATE = optimizer.param_groups[0]['lr']
#     # lr_cooldown = 0
#
#     while (game < config.GAMES):
#
#         print('----- GAME : ',game, '/', config.GAMES ,'------')
#
#         ######################### Learner Plays #########################
#
#         print('*Learner*')
#         print('mean/min/max sample weights : ,',np.round(np.mean(sample_weights),precision),
#               np.round(np.min(sample_weights),precision),np.round(np.max(sample_weights),precision))
#
#         if config.modality == 'impweight': #this is default modality
#             print('- mode : impweight')
#             pd_train[config.weights_tag] = sample_weights/np.mean(sample_weights)
#             train_dataloader = dataloader_functional(pd_train, weights_tag=config.weights_tag,
#                                                      sampler_on=False, sampler_tag=config.sampler_tag)
#         if config.modality == 'sampler':
#             print('-  mode : sampler')
#             pd_train[config.weights_tag] = 1.0  # importance weight set to 1, sampler is sampling according to sample_weights
#             pd_train[config.sampler_tag] = sample_weights/np.sum(sample_weights)
#             train_dataloader = dataloader_functional(pd_train, weights_tag=config.weights_tag,
#                                                      sampler_on=True, sampler_tag=config.sampler_tag)
#
#         if (game == 0) & (config.EPOCHS_LEARNER_WARMUP>config.EPOCHS_LEARNER):
#             EPOCHS_LEARNER = config.EPOCHS_LEARNER_WARMUP
#         else:
#             EPOCHS_LEARNER = config.EPOCHS_LEARNER
#
#         for epoch in range(config.EPOCHS_LEARNER):
#
#             history['learning_rate_learner'].append(optimizer.param_groups[0]['lr'] + 0)
#
#             output = BPF_epoch_learner(train_dataloader, classifier_network, criterion, optimizer, config.DEVICE, metrics_dic=metrics_dic,
#                               reg_dic=reg_dic, reg_weights=reg_weights, train_type=True)
#
#             history['loss_a1_mbatch_train'].append(output['loss'])
#             history['loss_a0_mbatch_train'].append(output['comp_loss'])
#             history['criterion_mbatch_train'].append(output['criterion'])
#
#             if reg_dic is not None:
#                 for reg in reg_dic.keys():
#                     history[reg+'_mbatch_train'].extend(output[reg])
#
#             if metrics_dic is not None:
#                 for metric in metrics_dic.keys():
#                     history[metric + '_mbatch_train'].extend(output[metric])
#         #####################################################
#
#
#         ########## Evaluation on worst case partition (train and validation) ############
#         output_train = epoch_persample_evaluation(train_eval_dataloader, classifier_network, criterion, config.DEVICE, metrics_dic=metrics_dic)
#         output_val = epoch_persample_evaluation(val_eval_dataloader, classifier_network, criterion, config.DEVICE,metrics_dic=metrics_dic)
#
#         weight_worst_train = BPF_projector.get_worst_weights(np.array(output_train['loss']))
#         weight_worst_val  = BPF_projector.get_worst_weights(np.array(output_val['loss']))
#
#         print('Sanity check: ')
#         print('mean/min/max worst_weights_train : ,',np.round(np.mean(weight_worst_train),precision),
#               np.round(np.min(weight_worst_train),precision),np.round(np.max(weight_worst_train),precision))
#         print('mean/min/max worst_weights_val : ,', np.round(np.mean(weight_worst_val), precision),
#               np.round(np.min(weight_worst_val), precision), np.round(np.max(weight_worst_val), precision))
#         print()
#
#         #train worst partition losses
#         den_a1_train = np.mean(weight_worst_train) if np.mean(weight_worst_train)>0 else 1e-20 #denominator
#         den_a0_train = (BPF_projector.upper-den_a1_train) if BPF_projector.upper-den_a1_train>0 else 1e-20
#
#         loss_a1_train = np.round(np.mean(weight_worst_train*np.array(output_train['loss']))/den_a1_train,precision)
#         loss_a0_train = np.round(np.mean(np.abs(BPF_projector.upper-weight_worst_train) * np.array(output_train['loss']))/den_a0_train,precision) #division by 0
#         average_loss_train = np.round(np.mean(np.array(output_train['loss'])), precision)
#
#         # val worst partition losses
#         den_a1_val = np.mean(weight_worst_val) if np.mean(weight_worst_val) > 0 else 1e-20 #denominator
#         den_a0_val = (BPF_projector.upper - den_a1_val) if BPF_projector.upper - den_a1_val > 0 else 1e-20
#
#         loss_a1_val = np.round(np.mean(weight_worst_val * np.array(output_val['loss']))/den_a1_val,precision)
#         loss_a0_val = np.round(np.mean(np.abs(BPF_projector.upper - weight_worst_val) * np.array(output_val['loss']))/den_a0_val,precision)
#         average_loss_val = np.round(np.mean(np.array(output_val['loss'])),precision)
#
#         ###############  Regulator Plays (new weight update) ###############
#         print('*Regulator*')
#
#         weight_list, _, _, flag_out = BPF_projector.update_weights(np.array(output_train['loss']), sample_weights,
#                                                                    iterations=config.EPOCHS_REGULATOR)
#         sample_weights = weight_list[-1]
#         loss_iw_a1_train = np.round(
#             np.mean(sample_weights * np.array(output_train['loss'])) / np.maximum(np.mean(sample_weights), 1e-20),
#             precision)
#         loss_iw_a0_train = np.round(
#             np.mean(np.abs(BPF_projector.upper - sample_weights) * np.array(output_train['loss'])) / np.maximum(
#                 np.mean(np.abs(BPF_projector.upper - sample_weights)), 1e-20), precision)  # division by 0
#
#         #### Saving worst case losses..
#         history['delta_weights_train'].extend([np.sum(np.abs(weight_worst_train - sample_weights))/np.sum(sample_weights) for i in range(EPOCHS_LEARNER)])
#
#         #first save train loss with the current sample weights!!!
#         history['loss_iw_a1_train'].extend([loss_iw_a1_train for i in range(EPOCHS_LEARNER)])
#         history['loss_iw_a0_train'].extend([loss_iw_a0_train for i in range(EPOCHS_LEARNER)])
#         history['loss_a1_train'].extend([loss_a1_train for i in range(EPOCHS_LEARNER)])
#         history['loss_a0_train'].extend([loss_a0_train for i in range(EPOCHS_LEARNER)])
#         history['loss_average_train'].extend([average_loss_train for i in range(EPOCHS_LEARNER)])
#         history['loss_a1_val'].extend([loss_a1_val for i in range(EPOCHS_LEARNER)])
#         history['loss_a0_val'].extend([loss_a0_val for i in range(EPOCHS_LEARNER)])
#         history['loss_average_val'].extend([average_loss_val for i in range(EPOCHS_LEARNER)])
#
#         #metrics
#         if metrics_dic is not None:
#             for metric in metrics_dic.keys():
#                 worst_metric_a1_train = np.mean(weight_worst_train * np.array(output_train[metric]))/den_a1_train
#                 worst_metric_a0_train = np.mean((BPF_projector.upper - weight_worst_train) * np.array(output_train[metric]))/den_a0_train
#                 average_metric_train = np.mean(np.array(output_train[metric]))
#
#                 history[metric + '_a1_train'].extend([np.round(worst_metric_a1_train,precision) for i in range(EPOCHS_LEARNER)])
#                 history[metric + '_a0_train'].extend([np.round(worst_metric_a0_train,precision) for i in range(EPOCHS_LEARNER)])
#                 history[metric + '_average_train'].extend([np.round(average_metric_train, precision) for i in range(EPOCHS_LEARNER)])
#
#                 worst_metric_a1_val = np.mean(weight_worst_val * np.array(output_val[metric]))/den_a1_val
#                 worst_metric_a0_val = np.mean((BPF_projector.upper - weight_worst_val) * np.array(output_val[metric]))/den_a0_val
#                 average_metric_val = np.mean(np.array(output_val[metric]))
#
#                 history[metric + '_a1_val'].extend([np.round(worst_metric_a1_val, precision) for i in range(EPOCHS_LEARNER)])
#                 history[metric + '_a0_val'].extend([np.round(worst_metric_a0_val, precision) for i in range(EPOCHS_LEARNER)])
#                 history[metric + '_average_val'].extend([np.round(average_metric_val, precision) for i in range(EPOCHS_LEARNER)])
#
#         history['game'].extend([game for _ in np.arange(EPOCHS_LEARNER)])
#         history['epoch'].extend([_ for _ in np.arange(EPOCHS_LEARNER)])
#         history['learning_rate_regulator'].extend([BPF_projector.eta for _ in np.arange(EPOCHS_LEARNER)])
#
#         print('-Learner summary ')
#
#         # save last model
#         model_params_save(config.basedir + config.model_name + '/' + config.last_model, classifier_network, optimizer)
#         print('-----------')
#         print()
#
#         # Best model, stopping criteria, and print
#         if val_stopper:
#             primary_loss = np.maximum(loss_a1_val,loss_a0_val)
#             secondary_loss = np.minimum(loss_a1_val,loss_a0_val)
#         else:
#             primary_loss = np.maximum(loss_a1_train,loss_a0_train)
#             secondary_loss = np.minimum(loss_a1_train,loss_a0_train)
#
#         if (primary_loss < best_primary_loss) | ((primary_loss == best_primary_loss)&(secondary_loss<best_secondary_loss)) | (game == 0):
#             print('saving best model...')
#             model_params_save(config.basedir + config.model_name + '/' + config.best_model, classifier_network,optimizer)
#             best_game = game + 0
#             best_primary_loss = primary_loss + 0
#             best_secondary_loss = secondary_loss + 0
#             patience = 0
#
#             #save weights of best model
#             history['weight_worst_train'] = weight_worst_train
#             history['weight_worst_val'] = weight_worst_val
#             history['sample_weights'] = sample_weights
#             history['sample_train_loss'] = output_train['loss']
#             history['sample_val_loss'] = output_val['loss']
#
#         else:
#             patience += 1
#
#         # Print -----
#         string_print = 'Game: ' + str(game) + '; lr_L: ' + str(optimizer.param_groups[0]['lr']) + '; lr_R: ' + str(BPF_projector.eta) + '; weights_rho : ' +\
#                        str(np.mean(sample_weights))
#         string_print = string_print  + '| worst/best_iwloss_tr ' + str(np.round(loss_iw_a1_train, 3)) + ', ' + str(np.round(loss_iw_a0_train, 3)) +\
#                        '| worst_loss_tr/val ' + str(np.round(loss_a1_train, 3)) + ', ' + str(np.round(loss_a1_val, 3)) + \
#                        '| best_loss_tr/val ' + str(np.round(loss_a0_train, 3)) + ', ' + str(np.round(loss_a0_val, 3)) + \
#                        '| avg_loss_tr/val ' + str(np.round(average_loss_train, 3)) + ', ' + str(np.round(average_loss_val, 3))
#
#         string_print = string_print + '|primary current/best ' + str(np.round(primary_loss, precision)) + ', ' + str(np.round(best_primary_loss, precision)) + \
#                       '|secondary current/best ' + str(np.round(secondary_loss,precision)) + ', ' + str(np.round(best_secondary_loss, precision)) +\
#                       '| dif_weights : ' + str(np.round(np.sum(np.abs(weight_worst_train-sample_weights))/np.sum(sample_weights),precision))
#
#
#         string_print = string_print + '|best game :'+str(best_game)+', patience : ',str(patience)
#
#         print(string_print)
#         print()
#         print()
#
#         # consider printing constraints ?,
#         game += 1
#
#         if (patience > config.patience) & (game > warmup):
#             break
#
#     # -------- END TRAINING --------#
#
#     # load best network
#     history['best_game'] = best_game
#     print('Training Ended, loading best model : ', config.basedir + config.model_name + '/'+config.best_model)
#     model_params_load(config.basedir + config.model_name + '/'+config.best_model, classifier_network, optimizer, config.DEVICE)
#
#     return history
