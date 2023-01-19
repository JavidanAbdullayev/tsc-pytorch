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
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import torch.backends.cudnn as cudnn
import torch.optim as optim


def fit(classifier, trainloader, valloader, input_shape, nb_classes, dataset_name, epochs):
    print('Hi from FCN!')
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
        epochs = 2000

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
        test(model)
        if min_train_loss  > train_loss:
            min_train_loss = train_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()
    
        final_loss.append(train_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])

    # Training function end

    # Save results
    output_directory = os.path.abspath(os.getcwd()) + '/results_fcn/'
    if os.path.exists(output_directory) == False:
        os.mkdir(output_directory)

    output_directory = output_directory + dataset_name + '/'
    if os.path.exists(output_directory) == False:
        os.mkdir(output_directory)
    else:
        print('DONE')        


    torch.save(best_model_wts, output_directory +  'best_teacher_model.pt')


    # Save Logs
    duration = time.time() - start_time
    best_teacher = classifier(input_shape, nb_classes)
    best_teacher.load_state_dict(best_model_wts)
    best_teacher.cuda()
    
    print('Best Model Accuracy in below ')
    start_test_time = time.time()
    test(best_teacher)
    test_duration = time.time() - start_test_time

    print(test(best_teacher))

    df = pd.DataFrame(list(zip(final_loss, learning_rates)), columns =['loss', 'learning_rate'])
    index_best_model = df['loss'].idxmin()
    row_best_model = df.loc[index_best_model]
    df_best_model = pd.DataFrame(list(zip([row_best_model['loss']], [index_best_model+1])), columns =['best_model_train_loss', 'best_model_nb_epoch'])

    df.to_csv(output_directory + 'history.csv', index=False)
    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    loss_, acc_ = test(best_teacher)
    df_metrics = pd.DataFrame(list(zip([min_train_loss], [acc_], [duration], [test_duration])), columns =['Loss', 'Accuracy', 'Duration', 'Test Duration'])
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)



def Classifiers(datasets_dict, dataset_names, classifier_names, epochs=-1, batch_size=-1):
    print('\n\n\n\n')
    print('Dataset names: ', dataset_names)
    print('Classifier name: ', classifier_names)
    print('\n\n\n\n')

    for dataset_name in dataset_names:
        for classifier_name in classifier_names:
            print('Classifier Type: ', classifier_name)   
            trainloader, valloader, input_shape, nb_classes = prepare_data(datasets_dict, dataset_name, classifier_name, batch_size)
            
            if classifier_name == 'fcn':
                from test_tscai_1.models import fcn
                print('Everything is ok and now FCN is calling')
                fcn.fit(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs, )

            elif classifier_name == 'cnn':
                from test_tscai_1.models import cnn
                print('Everything is ok and now CNN is calling')
                cnn.fit(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs)

            elif classifier_name == 'mlp':
                from test_tscai_1.models import mlp
                print('Everything is ok and now MLP is calling')
                mlp.fit(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs)

            elif classifier_name == 'resnet':
                from test_tscai_1.models import resnet
                print('Everything is ok and now RESNET is calling')
                resnet.fit(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs)

            elif classifier_name == 'inception':
                from test_tscai_1.models import inception
                print('Everything is ok and now INCEPTION is calling')
                inception.fit(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs)