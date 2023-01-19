import torch
from torch import nn

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
