import numpy as np
import sys
sys.path.append("../")
from general.utils import early_stopping,model_params_save,model_params_load,TravellingMean,to_np,save_json
import torch.nn.functional as F
import torch
import argparse

def  DRO_epoch_training(dataloader, classifier_network, criterions, eta, optimizer, DEVICE, train_type= 'train',metrics_dic = None):
    '''
    This function train or evaluates an epoch
    #inputs:
    dataloader, optimizer, classifier_network
    criterions: function that provides a base_loss
    logger: (tensorboard)
    train_type: if train performs backprop y otherwise only evaluates
    #Outputs:
    base_loss_all_out: output base loss per sensitive
    accuracy_out: output accuracy per sensitive
    full_loss_out: output full loss
    '''

    ###    INITIALIZE MEAN OBJECTS  #######
    #loss summary lists
    output = {}
    output['criterion'] = TravellingMean()
    output['loss'] = TravellingMean()

    if metrics_dic is not None: #additional metrics
        for key in metrics_dic.keys():
            output[key] = TravellingMean()

    if train_type.lower() == 'train':
        classifier_network = classifier_network.train()
    else:
        classifier_network = classifier_network.eval()

    # Loop through samples
    for i_batch, sample_batch in enumerate(dataloader):
        x, utility = sample_batch
        x = x.to(DEVICE)
        utility = utility.to(DEVICE)  # batch x nutility

        # zero the parameter gradients
        optimizer.zero_grad()

        # get output and losses
        logits = classifier_network(x)
        base_loss = criterions(logits, utility)

        dro_loss = F.relu(base_loss-eta)+eta


        if train_type.lower() == 'train':
            # classifier backpropagation
            dro_loss.mean().backward()
            optimizer.step()

        base_loss_np = to_np(base_loss)
        dro_loss_np = to_np(dro_loss)
        mass = utility.shape[0] #number samples batch

        output['criterion'].update(val=base_loss_np,mass=mass)  # base loss
        output['loss'].update(val=dro_loss_np,mass=mass) #dro loss!

        if metrics_dic is not None:
            for key in metrics_dic.keys():
                metric = metrics_dic[key](logits, utility)
                output[key].update(np.array([to_np(metric)],mass=mass))

    ################ Final Epoch performance ######################################
    for key in output.keys(): #return only means
        output[key] = output[key].mean


    return output

def DRO_trainer(train_dataloader, val_dataloader,
                optimizer, classifier_network, criterion, config,
                val_stopper=True, metric_stopper = 'loss',metrics_dic = None, precision = 3):

    # from config uses : config.EPOCHS, config.device, config.patience,config.EPOCHS_WARMUP
    tag_print = ['loss','criterion']

    history={}
    history['epoch'] = []
    history['learning_rate'] = []

    for tag in tag_print:
        history[tag + '_train'] = []
        history[tag + '_val'] = []

    if metrics_dic is not None:
        for metric in metrics_dic.keys():
            history[metric + '_train'] = []
            history[metric + '_val'] = []
            tag_print.append(metric)


    # if val_stopper:
    #     metric_stopper =  metric_stopper + '_val'
    # else:
    #     metric_stopper = metric_stopper + '_train'



    stop = False
    epoch = 0

    best_epoch = {'train': 0, 'val': 0}
    stopper = early_stopping(config.patience, 0, np.infty)
    stopper_train = early_stopping(config.patience, 0, np.infty)

    while (not stop) & (epoch < config.EPOCHS):

        # save learning rate
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        history['epoch'].append(epoch)

        # save loss
        output_train = DRO_epoch_training(dataloader=train_dataloader, classifier_network=classifier_network,
                                      eta=config.eta, criterions=criterion, DEVICE=config.DEVICE,
                                      optimizer=optimizer, train_type='train',metrics_dic=metrics_dic)

        output_val = DRO_epoch_training(dataloader=val_dataloader, classifier_network=classifier_network,
                                      eta=config.eta, criterions=criterion, DEVICE=config.DEVICE,
                                      optimizer=optimizer, train_type='val',metrics_dic=metrics_dic)

        ## history update
        for key in output_train.keys():
            history[key+'_train'].append(np.round(output_train[key],precision))
            history[key+'_val'].append(np.round(output_val[key],precision))

        # model_params_save(config.basedir + config.model_name + '/' + config.last_model, classifier_network, optimizer) #save last model

        ### bes Validation
        save_val, stop_val = stopper.evaluate(history[metric_stopper + '_val'][-1])
        if save_val:
            best_epoch['val'] = epoch
            print('saving best model, epoch: ', epoch)
            model_params_save(config.basedir + config.model_name + '/' + config.best_model, classifier_network,
                              optimizer)  # save best model

        ### best Train
        save_train, stop_train = stopper_train.evaluate(history[metric_stopper + '_train'][-1])
        if save_train:
            best_epoch['train'] = epoch
            print('saving best model, epoch: ', epoch)
            model_params_save(config.basedir + config.model_name + '/' + config.best_model_train, classifier_network,
                              optimizer) # save best model train

        if val_stopper:
            stop = stop_val
        else:
            stop = stop_train

        if  epoch < config.EPOCHS_WARMUP:
            stop = False #stop = false if we are still in the warmup epochs

        # print
        if ((epoch % config.n_print == config.n_print - 1) & (epoch >= 1)) | (epoch == 0):
            string_print = 'Epoch: ' + str(epoch) + '; lr: ' + str(optimizer.param_groups[0]['lr'])
            for tag in tag_print:
                string_print = string_print + ' |' + tag + '_tr/val : ' + str(history[tag+'_train'][-1]) + ',' +str(history[tag+'_val'][-1])
            print(string_print+' |best loss (tr,val) : ' + str(history[metric_stopper + '_train' ][best_epoch['train']]) +\
                  ',' + str(history[metric_stopper + '_val' ][best_epoch['val']]) + ' , stop_c : ' + str(stopper.counter))

        history['best_epoch'] = best_epoch
        epoch += 1

    # -------- END TRAINING --------#

    # load best network
    # print('Training Ended')
    # # model_params_save(config.best_network_path, classifier_network, optimizer)
    print('Training Ended, loading best model : ', config.basedir + config.model_name + '/' + config.best_model)
    model_params_load(config.basedir + config.model_name + '/' + config.best_model, classifier_network, optimizer, config.DEVICE)

    return history

class DRO_config(argparse.Namespace):

    def __init__(self,n_utility=2,**kwargs):

        self.basedir = 'models/'
        self.model_name = 'vanilla_model'
        self.best_model = 'weights_best.pth'
        self.best_model_train = 'weights_best_train.pth'

        self.seed = 42
        self.GPU_ID = 0

        self.BATCH_SIZE = 32
        self.n_workers = 32
        self.n_utility = n_utility
        self.augmentations = False
        self.resize = None

        ## network
        self.n_features = None
        self.hidden_layers = None
        self.batchnorm = False
        self.regression = False
        self.type_loss = 'L2'
        self.type_metric = []

        # Loss  -> todo!: I have to add regularizations

        ## Game ##
        self.EPOCHS = 1
        self.EPOCHS_WARMUP = 1
        self.patience = 15
        self.val_stopper = True
        self.n_print = 1

        self.LEARNING_RATE = 1e-5
        self.optim_weight_decay = 0
        self.optimizer = 'adam'
        self.lrdecay = 1

        ## DRO specific
        self.eta = 0.0 #learning rate regulator

        ## Load parameters
        for k in kwargs:
            setattr(self, k, kwargs[k])

        # Device
        torch.manual_seed(self.seed)
        if torch.cuda.is_available() and self.GPU_ID >= 0:
            DEVICE = torch.device('cuda:%d' % (self.GPU_ID))
        else:
            DEVICE = torch.device('cpu')
        self.DEVICE = DEVICE


        # def is_valid(self, return_invalid=False):
        # Todo! I have to update this properly
        # ok = {}
        #
        # if return_invalid:
        #     return all(ok.values()), tuple(k for (k, v) in ok.items() if not v)
        # else:
        #     return all(ok.values())

    def update_parameters(self, allow_new=True, **kwargs):
        if not allow_new:
            attr_new = []
            for k in kwargs:
                try:
                    getattr(self, k)
                except AttributeError:
                    attr_new.append(k)
            if len(attr_new) > 0:
                raise AttributeError("Not allowed to add new parameters (%s)" % ', '.join(attr_new))
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def save_json(self,save_path=None):
        config_dict = self.__dict__
        config2json = {}
        for key in config_dict.keys():
            if key != 'DEVICE':
                if type(config_dict[key]) is np.ndarray:
                    config2json[key] = config_dict[key].tolist()
                else:
                    config2json[key] = config_dict[key]
        if save_path is None:
            save_path =  self.basedir + self.model_name + '/config.json'
        save_json(config2json, save_path)
        print('Saving config json file in : ', save_path)

