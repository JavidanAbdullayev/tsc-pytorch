import torch
from torch import nn

class conv_block(nn.Module):

  def __init__(self, in_channels, out_channels, **kwargs):
    super(conv_block, self).__init__()
    self.relu = nn.ReLU()
    self.bathcnorm = nn.BatchNorm1d(out_channels)
    self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)


  def forward(self, x):
    return self.relu(self.bathcnorm(self.conv(x)))


class Classifier_FCN(nn.Module):

    def __init__(self, input_shape, nb_classes):
        super(Classifier_FCN, self).__init__()
        self.nb_classes = nb_classes

        self.conv1 = conv_block(in_channels=1, out_channels=128, kernel_size=8,  stride=1, padding='same')
        self.conv2 = conv_block(in_channels=128, out_channels=256, kernel_size=5,  stride=1, padding='same')
        self.conv3 = conv_block(in_channels=256, out_channels=128, kernel_size=3,  stride=1, padding='same')

        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, self.nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.avgpool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


