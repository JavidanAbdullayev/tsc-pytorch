import pandas as pd
import numpy as np
import sklearn
import os
from importlib import resources
from utils import transform_labels, read_all_datasets_fake, prepare_data
import torch
from torch.utils.data import DataLoader, TensorDataset

dataset_names = ['ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken']
datasets_dict = read_all_datasets_fake(dataset_names)

classifier_names = ['cnn', 'inception']
epochs=2
batch_size=-1
print('\n\n\n\n')
print('Dataset names: ', dataset_names)
print('Classifier name: ', classifier_names)
print('\n\n\n\n')

for dataset_name in dataset_names:
    for classifier_name in classifier_names:
        print('Classifier Type: ', classifier_name)   
        trainloader, valloader, input_shape, nb_classes = prepare_data(datasets_dict, dataset_name, classifier_name, batch_size)
        
        if classifier_name == 'fcn':
            from models import fcn
            print('Everything is ok and now FCN is calling')
            fcn.fit(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs)

        elif classifier_name == 'cnn':
            from models import cnn
            print('Everything is ok and now CNN is calling')
            cnn.fit(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs)

        elif classifier_name == 'mlp':
            from models import mlp
            print('Everything is ok and now MLP is calling')
            mlp.fit(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs)

        elif classifier_name == 'resnet':
            from models import resnet
            print('Everything is ok and now RESNET is calling')
            resnet.fit(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs)

        elif classifier_name == 'inception':
            from models import inception
            print('Everything is ok and now INCEPTION is calling')
            inception.fit(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs)