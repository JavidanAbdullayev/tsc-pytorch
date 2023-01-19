import pandas as pd
import numpy as np
import os
from importlib import resources
import sklearn
from sklearn.preprocessing import LabelEncoder
import torch
import test_datasets_1
from torch.utils.data import DataLoader, TensorDataset
from constants import *

def read_all_datasets(DATASET_NAMES_2018) -> pd.DataFrame:

    dataset_dict = test_datasets_1.DataLoader(DATASET_NAMES_2018)
    
    return dataset_dict

def prepare_data(datasets_dict, dataset_name, classifier_name, batch_size):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # make the min to zero of labels
    y_train, y_test = transform_labels(y_train, y_test)

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)
    y_true_train = y_train.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    new_x_train = np.transpose(x_train, (0, 2, 1))
    new_x_test = np.transpose(x_test, (0, 2, 1))
    
    x_train, y_train= torch.from_numpy(new_x_train), torch.from_numpy(y_train)
    x_test, y_test= torch.from_numpy(new_x_test), torch.from_numpy(y_test)
    input_shape = x_train.shape[1:]

    print('CLassifier name: ', classifier_name)
    if batch_size > 0:
        mini_batch_size = batch_size
    else:
        if classifier_name == 'fcn':
            mini_batch_size = int(min(x_train.shape[0]/10, 16))
        elif classifier_name == 'cnn':
            mini_batch_size = number_of_batches['cnn']
        elif classifier_name == 'mlp':
            mini_batch_size = int(min(x_train.shape[0]/10, 16))
        elif classifier_name == 'resnet':
            mini_batch_size = int(min(x_train.shape[0]/10, 16))
        elif classifier_name == 'inception':
            mini_batch_size = number_of_batches['inception']


    trainloader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=mini_batch_size,
        shuffle=True,
    )
    valloader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=mini_batch_size,
        shuffle=False
    )

    print('batch_size is ', mini_batch_size)
    return trainloader, valloader, input_shape, nb_classes


def transform_labels(y_train, y_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    # no validation split
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit
    y_train_test = np.concatenate((y_train, y_test), axis=0)
    # fit the encoder
    encoder.fit(y_train_test)
    # transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_train_test)
    # resplit the train and test
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]
    return new_y_train, new_y_test


def read_all_datasets_fake(DATASET_NAMES_2018) -> pd.DataFrame:
    datasets_dict = {}

    for dataset_name in DATASET_NAMES_2018:
        df_train =  pd.read_csv('/home/javidan/Desktop/TSC_in_Pytorch/test_tscai_1/data/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)
        df_test =  pd.read_csv('/home/javidan/Desktop/TSC_in_Pytorch/test_tscai_1/data/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

        y_train = df_train.values[:, 0]
        y_test = df_test.values[:, 0]
        
        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])
        
        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])
        
        x_train = x_train.values
        x_test = x_test.values
        
        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_
        
        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
        
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())


    return datasets_dict
