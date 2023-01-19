import pandas as pd
import numpy as np
import sklearn
import copy
import time
import os
from importlib import resources
from test_tscai_1.utils import transform_labels, read_all_datasets, prepare_data
import torch
from torch.utils.data import DataLoader, TensorDataset

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import torch.backends.cudnn as cudnn
import torch.optim as optim
from test_tscai_1.constants import *



def fit(classifier, trainloader, valloader, input_shape, nb_classes, dataset_name, classifier_name, epochs, validation_data=False):
    model = classifier(input_shape, nb_classes)
    print(model)

    use_cuda = torch.cuda.is_available()                
    if use_cuda:
        torch.cuda.set_device(0)
        model.cuda()
        cudnn.benchmark = True
    summary(model, ( input_shape[-2], input_shape[-1]))
    
    # Training
    def train_alone_model(net, epoch):

        print('\Teaining epoch: %d' % epoch)
        net.train()

        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()

            outputs = net(inputs.float())
            loss = criterion_CE(outputs, targets)
            loss.backward() 
            optimizer.step()  

            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float().item()

            b_idx = batch_idx

        print('Training Loss: %.3f | Training Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

        return train_loss / (b_idx + 1)

    def test(net):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(valloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs.float())
            loss = criterion_CE(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float().item()
            b_idx = batch_idx

        print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
        return test_loss / (b_idx + 1), correct / total


    final_loss = []
    learning_rates = []
    
    if epochs < 0:
        epochs = number_of_epochs[classifier_name]
    
    criterion_CE = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=epochs, steps_per_epoch=len(trainloader))

    use_cuda = torch.cuda.is_available()        
    best_model_wts = copy.deepcopy(model.state_dict())
    min_train_loss = np.inf

    # Training function start
    start_time = time.time()    
    for epoch in range(epochs):
        train_loss = train_alone_model(model, epoch)
        if validation_data:
            test(model)

        if min_train_loss  > train_loss:
            min_train_loss = train_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()
    
        final_loss.append(train_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])

    # Training function end


    # Save results
    output_directory = os.path.abspath(os.getcwd()) + '/results_' + classifier_name + '/'
    if os.path.exists(output_directory) == False:
        os.mkdir(output_directory)

    output_directory = output_directory + dataset_name + '/'
    
    if os.path.exists(output_directory) == False:
        os.mkdir(output_directory)
    else:
        print('DONE')        


    torch.save(best_model_wts, output_directory +  'best_model.pt')


    # Save Logs
    duration = time.time() - start_time
    best_model = classifier(input_shape, nb_classes)
    best_model.load_state_dict(best_model_wts)
    best_model.cuda()
    
    print('Best Model Accuracy in below ')
    start_test_time = time.time()
    test(best_model)
    test_duration = time.time() - start_test_time

    print(test(best_model))

    df = pd.DataFrame(list(zip(final_loss, learning_rates)), columns =['loss', 'learning_rate'])
    index_best_model = df['loss'].idxmin()
    row_best_model = df.loc[index_best_model]
    df_best_model = pd.DataFrame(list(zip([row_best_model['loss']], [index_best_model+1])), columns =['best_model_train_loss', 'best_model_nb_epoch'])

    df.to_csv(output_directory + 'history.csv', index=False)
    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    loss_, acc_ = test(best_model)
    df_metrics = pd.DataFrame(list(zip([min_train_loss], [acc_], [duration], [test_duration])), columns =['Loss', 'Accuracy', 'Duration', 'Test Duration'])
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)




def Classifiers(datasets_dict, classifier_names, epochs=-1, batch_size=-1, validation_data=False):

    dataset_names = list(datasets_dict.keys())
    
    for dataset_name in dataset_names:
        for classifier_name in classifier_names:
            trainloader, valloader, input_shape, nb_classes = prepare_data(datasets_dict, dataset_name, classifier_name, batch_size)
            
            if classifier_name == 'fcn':
                from test_tscai_1.models import fcn
                fit(fcn.Classifier_FCN, trainloader, valloader, input_shape, nb_classes, dataset_name, classifier_name, epochs, validation_data=validation_data)

            elif classifier_name == 'cnn':
                from test_tscai_1.models import cnn
                fit(cnn.Classifier_CNN, trainloader, valloader, input_shape, nb_classes, dataset_name, classifier_name, epochs, validation_data=validation_data)

            elif classifier_name == 'mlp':
                from test_tscai_1.models import mlp
                fit(mlp.Classifier_MLP, trainloader, valloader, input_shape, nb_classes, dataset_name, classifier_name, epochs, validation_data=validation_data)

            elif classifier_name == 'resnet':
                from test_tscai_1.models import resnet
                fit(resnet.Classifier_RESNET, trainloader, valloader, input_shape, nb_classes, dataset_name, classifier_name, epochs, validation_data=validation_data)

            elif classifier_name == 'inception':
                from test_tscai_1.models import inception
                fit(inception.Classifier_INCEPTION, trainloader, valloader, input_shape, nb_classes, dataset_name, classifier_name, epochs, validation_data=validation_data)
