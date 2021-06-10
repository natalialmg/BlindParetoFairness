import numpy as np
import torch
import json
from pathlib import Path

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def model_params_save(filename,classifier_network, optimizer):
    torch.save([classifier_network.state_dict(),optimizer.state_dict()], filename)

def model_params_load(filename,classifier_network, optimizer,DEVICE):
    classifier_dic, optimizer_dic = torch.load(filename, map_location=DEVICE)
    classifier_network.load_state_dict(classifier_dic)
    optimizer.load_state_dict(optimizer_dic)

def load_json(fpath):
    with open(fpath,'r') as f:
        return json.load(f)

def save_json(data,fpath,**kwargs):
    with open(fpath,'w') as f:
        f.write(json.dumps(data,**kwargs))

def to_np(t):
    return t.cpu().detach().numpy()

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical



class early_stopping(object):
    def __init__(self, patience, counter, best_loss):
        self.patience = patience #max number of nonimprovements until stop
        self.counter = counter #number of consecutive nonimprovements
        self.best_loss = best_loss

    def evaluate(self, loss):
        save = False #save nw
        stop = False #stop training
        if loss < 0.999*self.best_loss:
            self.counter = 0
            self.best_loss = loss
            save = True
            stop = False
        else:
            self.counter += 1
            if self.counter > self.patience:
                stop = True

        return save, stop

class TravellingMean:
    def __init__(self):
        self.count = 0
        self._mean= 0

    @property
    def mean(self):
        return self._mean

    def update(self, val, mass=None):
        if mass is None:
            mass = val.shape[0]
        self.count+=mass
        self._mean += ((np.mean(val)-self._mean)*mass)/self.count

    def __str__(self):
        return '{:.3f}'.format(self._mean)

    def __repr__(self):
        return '{:.3f}'.format(self._mean)


import time, os
from pynvml import *
from subprocess import Popen
import numpy as np
import pickle, shlex
nvmlInit()

def run_command(cmd, minmem=2,use_env_variable=True, admissible_gpus=[1],sleep=60):
    sufficient_memory = False
    gpu_idx=0

    while not sufficient_memory:
        time.sleep(sleep)
        frees=[]
        for gpu in admissible_gpus:
            info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(gpu))
            free = info.free / 1024 / 1024 / 1024
            frees.append(free)
        frees = np.array(frees)
        if not use_env_variable: #safe mode
            sufficient_memory = np.min(frees) >=minmem  # 4.5 Gb
        else:
            sufficient_memory = np.max(frees) >= minmem  # 4.5 Gb
        gpu_idx = admissible_gpus[np.argmax(frees)]

    if use_env_variable:
        # os.system('CUDA_VISIBLE_DEVICES="{}" '.format(gpu_idx) +cmd)
        proc = Popen(['CUDA_VISIBLE_DEVICES="{}" '.format(gpu_idx) + cmd.format(0)], shell=True,
                     stdin=None, stdout=None, stderr=None, close_fds=True)
        print('CUDA_VISIBLE_DEVICES="{}" '.format(gpu_idx) + cmd.format(0))
    else:
        proc = Popen(shlex.split(cmd.format(gpu_idx)), shell=False,
                     stdin=None, stdout=None, stderr=None, close_fds=True)
        print( cmd.format(gpu_idx))




