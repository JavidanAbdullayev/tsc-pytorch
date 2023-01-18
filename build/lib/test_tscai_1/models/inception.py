import math
import copy
import time
import numpy as np
import pandas as pd
import os

import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import torch.backends.cudnn as cudnn
import torch.optim as optim


class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual_block, self).__init__()
        
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = self.residual(x)

        return x

class Inception_block(nn.Module):

  def __init__(self, in_channels, out_channels, stride=1, use_residual=True, use_bottleneck=True):
    super(Inception_block, self).__init__()
    
    self.use_bottleneck = use_bottleneck
    self.in_channels = in_channels
    self.out_channels = out_channels 
    self.bottleneck_size = 32


    if self.in_channels == 1:
        self.in_channels = int(in_channels)
    else:
        self.in_channels = int(self.bottleneck_size)
        self.bottleneck =  nn.Conv1d(in_channels, self.bottleneck_size, kernel_size=1, padding='same', bias=False)

    self.branch1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=40,  stride=1, padding='same', bias=False)
    self.branch2 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=20,  stride=1, padding='same', bias=False)
    self.branch3 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=10,  stride=1, padding='same', bias=False)

    self.branch4 = nn.Sequential(
        nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        nn.Conv1d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding='same', bias=False),
    )

    self.bn = nn.BatchNorm1d(4*self.out_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    if self.use_bottleneck and int(x.shape[-2]) > 1:
        y = self.bottleneck(x)
    else:
        y = x        

    x = torch.cat([self.branch1(y), self.branch2(y), self.branch3(y), self.branch4(x)], 1)
    x = self.bn(x)
    x = self.relu(x)

    return x

class Classifier_INCEPTION(nn.Module):

    def __init__(self, input_shape, nb_classes, nb_filters=32, depth=6, residual=True):
        super(Classifier_INCEPTION, self).__init__()
        self.nb_classes = nb_classes
        self.nb_filters = nb_filters
        self.residual = residual
        self.depth = depth

        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        
        for d in range(depth):
            self.inception.append(Inception_block(1 if d == 0 else nb_filters * 4, nb_filters))
            
            if self.residual and d % 3 == 2: 
                self.shortcut.append(Residual_block(1 if d == 2 else 4*nb_filters, 4*nb_filters))


        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(4*nb_filters, self.nb_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        input_res = x

        for d in range(self.depth):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2:
                y = self.shortcut[d//3](input_res)
                x = self.relu(x + y)
                input_res = x

        x = self.avgpool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x




def fit(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs):
    print('Hi from Inception!')
    recupereTeacherLossAccurayTest2 = []
    teacher = Classifier_INCEPTION(input_shape, nb_classes)

    print(teacher)

    use_cuda = torch.cuda.is_available()                
    if use_cuda:
        torch.cuda.set_device(0)
        teacher.cuda()
        cudnn.benchmark = True

    summary(teacher, ( input_shape[-2], input_shape[-1]))
    

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
        epochs = 1500

    criterion_CE = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001,)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=epochs, steps_per_epoch=len(trainloader))

    use_cuda = torch.cuda.is_available()        
    best_model_wts = copy.deepcopy(teacher.state_dict())
    min_train_loss = np.inf

    start_time = time.time()    
    for epoch in range(epochs):
        train_loss = train_alone_model(teacher, epoch)
        test(teacher)
        if min_train_loss  > train_loss:
            min_train_loss = train_loss
            best_model_wts = copy.deepcopy(teacher.state_dict())
        scheduler.step()
    
        final_loss.append(train_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])
    
    output_directory = os.path.abspath(os.getcwd()) + '/results_inception/'
    if os.path.exists(output_directory) == False:
        os.mkdir(output_directory)

    output_directory = os.path.abspath(os.getcwd()) + '/results_inception/' + dataset_name + '/'
    if os.path.exists(output_directory) == False:
        os.mkdir(output_directory)
    else:
        print('DONE')        

        
    torch.save(best_model_wts, output_directory +  'best_teacher_model.pt')


    # Save Logs
    duration = time.time() - start_time
    best_teacher = Classifier_INCEPTION(input_shape, nb_classes)
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

