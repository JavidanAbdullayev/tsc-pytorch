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


class conv_block(nn.Module):

  def __init__(self, in_channels, out_channels,  **kwargs):
    super(conv_block, self).__init__()
    self.relu = nn.ReLU()
    self.bathcnorm = nn.BatchNorm1d(out_channels)
    self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)


  def forward(self, x):
    return self.relu(self.bathcnorm(self.conv(x)))

class conv_block_2(nn.Module):

  def __init__(self, in_channels, out_channels,  **kwargs):
    super(conv_block_2, self).__init__()
    self.block = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, **kwargs),
        nn.BatchNorm1d(out_channels)
    )


  def forward(self, x):
    return self.block(x)


class Classifier_RESNET(nn.Module):

    def __init__(self, input_shape, nb_classes):
        super(Classifier_RESNET, self).__init__()
        self.nb_classes = nb_classes
        n_feature_maps = 64
        
        # BLOCK 1
        self.conv_x = conv_block(in_channels=1,                out_channels=n_feature_maps, kernel_size=8,  stride=1, padding='same')
        self.conv_y = conv_block(in_channels=n_feature_maps,   out_channels=n_feature_maps, kernel_size=5,  stride=1, padding='same')
        self.conv_z = conv_block_2(in_channels=n_feature_maps, out_channels=n_feature_maps, kernel_size=3, stride=1, padding='same')
        # expand channels for the sum
        self.shortcut_y = conv_block_2(in_channels=1, out_channels=n_feature_maps, kernel_size=1, stride=1, padding='same')
        self.relu = nn.ReLU()
    
        # BLOCK 2
        self.conv_x_2 = conv_block(in_channels=n_feature_maps,   out_channels=n_feature_maps*2, kernel_size=8,  stride=1, padding='same')
        self.conv_y_2 = conv_block(in_channels=n_feature_maps*2,   out_channels=n_feature_maps*2, kernel_size=5,  stride=1, padding='same')
        self.conv_z_2 = conv_block_2(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=3, stride=1, padding='same')
        # expand channels for the sum
        self.shortcut_y_2 = conv_block_2(in_channels=n_feature_maps, out_channels=n_feature_maps*2, kernel_size=1, stride=1, padding='same')
        self.relu = nn.ReLU()

        # BLOCK 3
        self.conv_x_3 = conv_block(in_channels=n_feature_maps*2,   out_channels=n_feature_maps*2, kernel_size=8,  stride=1, padding='same')
        self.conv_y_3 = conv_block(in_channels=n_feature_maps*2,   out_channels=n_feature_maps*2, kernel_size=5,  stride=1, padding='same')
        self.conv_z_3 = conv_block_2(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=3, stride=1, padding='same')
        # expand channels for the sum
        self.shortcut_y_3 = nn.BatchNorm1d(n_feature_maps*2)
        self.relu = nn.ReLU()


        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(2*n_feature_maps, self.nb_classes)

    def forward(self, x):
        input_layer = x

        # BLOCK 1
        conv_x = self.conv_x(x)
        conv_y = self.conv_y(conv_x)
        conv_z = self.conv_z(conv_y)
        shortcut_y = self.shortcut_y(input_layer)

        output_block_1 = shortcut_y + conv_z 
        output_block_1 = self.relu(output_block_1)
        # BLOCK 2
        conv_x_2 = self.conv_x_2(output_block_1)
        conv_y_2 = self.conv_y_2(conv_x_2)
        conv_z_2 = self.conv_z_2(conv_y_2)
        shortcut_y_2 = self.shortcut_y_2(output_block_1)
        output_block_2 = shortcut_y_2 + conv_z_2
        output_block_2 = self.relu(output_block_2)

        # BLOCK 3
        conv_x_3 = self.conv_x_3(output_block_2)
        conv_y_3 = self.conv_y_3(conv_x_3)
        conv_z_3 = self.conv_z_3(conv_y_3)
        shortcut_y_3 = self.shortcut_y_3(output_block_2)
        output_block_3 = shortcut_y_3 + conv_z_3
        output_block_3 = self.relu(output_block_3)

        x = self.avgpool1(output_block_3)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x




def fit(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs):
    print('Hi from RESNET!')
    model = Classifier_RESNET(input_shape, nb_classes)
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
        epochs = 1500
    criterion_CE = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=20, steps_per_epoch=len(trainloader))

    use_cuda = torch.cuda.is_available()        
    best_model_wts = copy.deepcopy(model.state_dict())
    min_train_loss = np.inf

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
    
    output_directory = os.path.abspath(os.getcwd()) + '/results_resnet/'
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
    best_teacher = Classifier_RESNET(input_shape, nb_classes)
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

