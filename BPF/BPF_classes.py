import argparse
import sys
sys.path.append("../")

import os
import torch
from torch import optim
from general.utils import save_json,mkdir,model_params_load
from general.losses import losses,metrics
from general.networks import VanillaNet, FCBody
import numpy as np
from general.evaluation import epoch_persample_evaluation

class BPF_projector():
    def __init__(self, **kwargs):
        self.epsilon = 0
        self.upper = 1
        self.rho = 1

        self.eta = 0.3
        self.proj_iterations = 1000
        self.proj_algo = None #is expected to be a function that

        for k in kwargs:
            print(k, kwargs[k]) #Todo!!: check what happens with epsilon
            setattr(self, k, kwargs[k])

        if self.proj_algo is None:
            # default is dykjstra
            self.proj_algo = l1_box_projection #lambda x: l1_box_projection(x,self.K,self.epsilon,self.upper)
            # self.proj_algo = minmaxTF_simplex_dykjstra_cyclic(self.ngroups,self.epsilon,self.nabla,
            #                                              self.K,iterations= self.proj_iterations)
        self.cost_delta_improve = 0.05
        self.max_weight_change = 0.05
        self.eta_increase_patience = 5
        self.decay = 0.75

        print('Valid Parameters : ',self.is_valid())

    def is_valid(self):
        valid = True

        ## Minimum size (epsilon) checks
        if self.epsilon<0:
            print('Warning!!: epsilon < 0')
            valid = False

        if self.epsilon > 1:
            print('Warning!!: epsilon > 1')
            valid = False

        if self.upper<0:
            print('Warning!!: upper < 0')
            valid = False

        if self.epsilon>self.upper:
            print('Warning!!: epsilon > upper')
            valid = False

        if self.rho<self.epsilon:
            print('Warning!!: rho < epsilon')
            valid = False

        if self.rho>self.upper:
            print('Warning!!: rho > upper')
            valid = False

        return valid
    def check_constrains(self,weight):
        c1 = np.round(np.min(weight-self.epsilon),3)
        c2 = np.round(np.min(self.upper - weight), 3)
        c3 = np.round(np.mean(weight),3) - self.rho

        # print('min(weight-epsilon) (>=0): ',c1)
        # print('min(upper-weight) (>=0): ', c2)
        # print('mean(weight) - rho (>=0): ',c3)

        return (c1>=0)&(c2>=0)&(c3>=0)

    def update_weights(self, loss, weight, iterations = 1, decay = None, verbose = True,
                       eta_increase_patience = None,cost_delta_improve = None, max_weight_change=None):

        if cost_delta_improve is None:
            cost_delta_improve = self.cost_delta_improve
        if max_weight_change is None:
            max_weight_change = self.max_weight_change
        if eta_increase_patience is None:
            eta_increase_patience = self.eta_increase_patience
        if decay is None:
            decay = self.decay

        weight_list = []
        cost_list = []
        eta_list = []
        flag_out = 'no_improvement'

        weight_list.append(np.array(weight))
        nweights = weight_list[-1].shape[0]
        loss = np.array(loss)
        cost_list.append(np.mean(weight*loss)/np.mean(weight))

        eta = self.eta + 0
        it = 0
        p_increase=0
        patience = 0
        # print(cost_list[-1])
        # print(weight)

        string_0 = 'eta_0 ' + str(eta)
        if (self.rho == self.upper): #only one feasible solution that is weight = self.upper since weights must sum self.K and are in [self.epsilon,self.upper]
            weight_i = self.upper*np.ones_like(weight)
            weight_list.append(weight_i)
            return weight_list, cost_list, eta_list,flag_out

        if (self.rho == self.epsilon): #only one feasible solution that is weight = self.epsilon since weights must sum self.K and are in [self.epsilon,self.upper]
            weight_i = self.epsilon*np.ones_like(weight)
            weight_list.append(weight_i)
            return weight_list, cost_list, eta_list,flag_out


        while (it < iterations):
            it += 1
            weight_i = self.proj_algo(weight_list[-1]+eta*loss,self.rho*nweights,
                                      self.epsilon*np.ones([nweights]),
                                      self.upper*np.ones([nweights]))
            # weight_i = weight_i/np.sum(weight_i) # guarantee that it is in the simplex

            if np.min(weight_i)<0:
                print('Warning: negative weight encountered')

            cost_i = np.mean(weight_i*loss)/np.mean(weight_i)
            # relative_improvement = (np.abs(cost_i - cost_list[0]) / cost_list[0])
            relative_improvement = (cost_i - cost_list[0]) / cost_list[0]
            relative_weight_improvement = np.sum(np.abs(weight_i-weight_list[0]))/np.sum(np.abs(weight_list[0]))

            # cost_i = np.sum(weight_i * loss)
            # cost_i = np.sum(weight_i * loss)/np.sum(weight_i)
            # print(cost_i,cost_list[0], cost_list[-1], relative_improvement)
            # print(weight_i)
            # print()

            if (cost_i <= cost_list[-1]):
                patience += 1
            else:
                patience = 0

            if patience == 20:
                string_print = string_0 + '; eta_T ' + str(eta) + \
                               '; terminate: no weight change {:.3e}'.format(cost_i-cost_list[-1])
                break  # enough improvement

            if cost_i < cost_list[-1]: #if step did not improve, decay eta and continue
                eta = eta*decay
                p_increase=0
                string_print = string_0 + '; eta_T ' + str(eta) + \
                               '; terminate: max iterations '
                # print('decrease')
                continue
            if (relative_improvement >1.2*cost_delta_improve) | (relative_weight_improvement >1.2*max_weight_change)  :
                eta = eta*decay
                p_increase=0
                string_print = string_0 + '; eta_T ' + str(eta) + \
                               '; terminate: max iterations '
                # print('decrease because too much improvement')
                continue

            p_increase +=1
            if p_increase>eta_increase_patience: #increase eta if we have seen improvements > patience
                # eta=eta/decay
                eta=eta*2/(decay+1)
                p_increase=0
                # print('increase')

            cost_list.append(cost_i)
            weight_list.append(weight_i)
            eta_list.append(eta)

            ####-stopping criteria-####

            #-enough improvement
            if relative_improvement > cost_delta_improve:
                # print('enough improvement ')
                string_print = string_0 + '; eta_T ' + str(eta) +\
                               '; terminate: enough cost improvement {:.3e}'.format(relative_improvement)
                flag_out = 'cost_delta_improve'
                break #enough improvement

            if relative_weight_improvement > max_weight_change:
                # print('enough improvement ')
                string_print = string_0 + '; eta_T ' + str(eta) +\
                               '; terminate: enough weight improvement {:.3e}'.format(relative_weight_improvement)
                flag_out = 'max_weight_change'
                break #enough improvement

            #-eta already too small
            if np.max(np.abs(eta*loss)) < 1e-20:
                # print('out low eta*loss ')
                string_print = string_0 + '; eta_T ' + str(eta) +\
                               '; terminate: low eta*loss {:.3e}'.format(np.max(np.abs(eta*loss)))
                flag_out = 'step_small'
                break


            #-iterations
            string_print = string_0 + '; eta_T ' + str(eta) + \
                           '; terminate: max iterations '

        if verbose:
            print(string_print)

        valid_constraint = self.check_constrains(weight_list[-1])
        print('valid_constraint:',valid_constraint)
        return weight_list,cost_list,eta_list,flag_out


    def get_worst_weights(self,loss):
        loss = np.array(loss)
        weights = np.ones(loss.shape[0])*self.epsilon

        budget = (self.rho - self.epsilon)*loss.shape[0]
        for ix in np.argsort(loss)[::-1]:
            budget_ix = np.minimum(budget,self.upper-self.epsilon)
            weights[ix] = weights[ix] + budget_ix
            budget -= budget_ix
            if budget <= 0:
                break
        print('valid weight: ',self.check_constrains(weights))
        return weights

class BPF_config(argparse.Namespace):

    def __init__(self,n_utility=2,**kwargs):

        self.basedir = 'models/'
        self.model_name = 'vanilla_model'
        self.best_model = 'weights_best.pth'
        self.best_model_train = 'weights_best_train.pth'
        # self.learner_model = 'weights_learner.pth'

        self.seed = 42
        self.GPU_ID = 0

        # self.balance_sampler = False
        self.weights_tag = 'weights'
        self.sampler_tag = 'weights_sampler' #in case the sampler option instead of IW is activated
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
        self.type_loss = 'CE'
        self.type_metric = []

        ## Game ##
        self.GAMES = 5
        self.GAMES_WARMUP = 30
        self.patience = 15
        self.val_stopper = False

        ## Learner ##
        self.EPOCHS_LEARNER = 1
        self.EPOCHS_LEARNER_WARMUP = 1
        self.LEARNING_RATE = 1e-5
        self.optimizer = 'adam'
        self.optim_weight_decay = 0
        self.n_print = 1
        self.modality = 'impweight' #options are going to be groups, sampler

        ## Regulator ##
        self.EPOCHS_REGULATOR = 500
        self.delta_improve_regulator = 0.05
        self.delta_weight_change_regulator = 0.05
        self.lrdecay = 0.75
        self.eta = 5 #learning rate regulator

        ## BPF Projector parameters ##
        self.epsilon = 0
        self.upper = 1
        self.rho = 1

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


class BPF_model():

    def __init__(self, config, classifier_network = None):
        self.config = config

        # mkdir(self.config.basedir)
        mkdir(os.path.join(self.config.basedir,self.config.model_name))


        #Make classifier network
        if classifier_network is None:
            self.classifier_network = VanillaNet(self.config.n_utility, body=FCBody(self.config.n_features,
                                                                          hidden_units=self.config.hidden_layers,
                                                                          batchnorm=self.config.batchnorm))
            self.classifier_network = self.classifier_network.to(self.config.DEVICE)
        else:
            self.classifier_network = classifier_network

        # Make optimizer
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(self.classifier_network.parameters(), lr=self.config.LEARNING_RATE,
                                   weight_decay=self.config.optim_weight_decay)
        elif self.config.optimizer == 'RMSprop':
                self.optimizer = optim.RMSprop(self.classifier_network.parameters(), lr=self.config.LEARNING_RATE,
                                          weight_decay=self.config.optim_weight_decay)
        else:
            print(' Loading SGD optimizer as default')
            self.optimizer = optim.SGD(self.classifier_network.parameters(), lr=self.config.LEARNING_RATE, momentum=0.9, nesterov=True,
                                       weight_decay=self.config.optim_weight_decay)

        ## Criterion
        self.criterion = losses(type_loss=self.config.type_loss,
                           regression=self.config.regression)

        self.metric_dic = None
        if len(self.config.type_metric) > 0:
            self.metric_dic = {}
            for metric_tag in self.config.type_metric:
                self.metric_dic[metric_tag] = metrics(type_loss=metric_tag)

        print('---------------- BPF model created ----------------------------')
        print()
        print('Model directory:', self.config.basedir + self.config.model_name + '/')
        print()
        print('-- Config file :')
        print(self.config)
        print('')
        print()
        print('-- Network : ')
        print(self.classifier_network)
        print()
        print()
        print('-- Optimizer : ')
        print(self.optimizer)
        print()
        print()
        mkdir(self.config.basedir)
        mkdir(self.config.basedir+self.config.model_name+'/')

        self.history = {}

    def load_network(self, load_file=None):
        # import os
        if os.path.exists(load_file):
            print(' Loading : ', load_file)
            model_params_load(load_file, self.classifier_network, self.optimizer, self.config.DEVICE)

    def train(self,pd_train, pd_val, dataloader_functional):



        ## Projector
        from BPF.BPF_classes import BPF_projector
        print('-- BPF Projector')
        max_projector = BPF_projector(eta=self.config.eta, rho=self.config.rho,
                                      upper=self.config.upper, epsilon=self.config.epsilon,
                                      cost_delta_improve=self.config.delta_improve_regulator,
                                      decay=self.config.lrdecay, max_weight_change=self.config.delta_weight_change_regulator)
        print()

        self.config.save_json()

        ## Trainer
        from BPF.BPF_trainers import BPF_trainer
        print('---------------------------- TRAINING ----------------------------')
        self.history = BPF_trainer(dataloader_functional, pd_train, pd_val,
                              self.optimizer, self.classifier_network, self.criterion, self.config, max_projector,metrics_dic = self.metric_dic,
                              val_stopper=self.config.val_stopper, warmup=self.config.GAMES_WARMUP)

        print(' Saving .... ')
        for key in self.history.keys():
            self.history[key] = np.array(self.history[key]).tolist()

        save_json(self.history, self.config.basedir + self.config.model_name + '/history.json')
        print('history file saved on : ', self.config.basedir + self.config.model_name + '/history.json')


        return self.history

    def eval(self,eval_dataloader):
        pd_eval = epoch_persample_evaluation(eval_dataloader, self.classifier_network, self.criterion, self.config.DEVICE,
                                   metrics_dic=self.metric_dic)
        return pd_eval


def l1_box_projection(y,c,e,u):



    def get_x(t):
        return np.minimum(np.maximum(e,y-t),u)

    def get_f(t):
        x = get_x(t)
        return np.sum(x) - c

    def get_sets(t):
        x = np.minimum(np.maximum(e, y - t), u)

        e_set = (x==e)
        u_set = (x==u)*(1-e_set) #make sure no overlapping
        t_set = 1-(e_set+u_set)

        return e_set,u_set,t_set

    iterate = 0
    t0 = np.max(y-e)
    t1 = np.min(y-u)

    ft0 = get_f(t0)
    ft1 = get_f(t1)

    if ft0 == 0:
        # print(iterate)
        return get_x(t0)
    if ft1 == 0:
        # print(iterate)
        return get_x(t1)


    while (iterate<5000):
        iterate += 1
        # print(ft0,ft1,t0,t1)

        th = (t0 + t1)/2
        fth = get_f(th)

        if (np.allclose(ft0, ft1, atol=1e-15) | np.allclose(t0, t1, atol=1e-15)):
            # print(iterate)
            return get_x(th)



        if (np.sign(fth)==np.sign(ft0)):
            t0 = th + 0
            ft0 = fth + 0
        elif (np.sign(fth)==np.sign(ft1)):
            t1 = th + 0
            ft1 = fth + 0
        else:
            # print(iterate)
            return get_x(th)

        # check consistency
        e_set, u_set, t_set = get_sets(th)

        if np.sum(t_set)>0:
            th_set = (np.sum(u_set*u) + np.sum(e_set*e) + np.sum(y*t_set) - c)/(np.sum(t_set))

            fth_set = get_f(th_set)

            if (np.sign(fth_set)==np.sign(ft0)):
                if (np.abs(th_set-t1) < np.abs(t0-t1)):  # tighter
                    t0 = th_set + 0
                    ft0 = fth_set + 0
            elif (np.sign(fth_set) == np.sign(ft1)):
                if (np.abs(th_set-t0) < np.abs(t1-t0)):  # tighter
                    t1 = th_set + 0
                    ft1 = fth_set + 0
            else:
                # print(iterate)
                return get_x(th_set)

    print('End due to max iterations, returning t, f(t) : ',(t0+t1)/2,get_f((t0+t1)/2))
    # print(iterate)
    return get_x((t0+t1)/2)



