
from torch import nn
import torch.nn.functional as F

class conv_block(nn.Module):

  def __init__(self, in_channels, out_channels, **kwargs):
    super(conv_block, self).__init__()

    self.block = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, **kwargs),
        nn.Sigmoid(),
        nn.AvgPool1d(kernel_size=3)
    )

  def forward(self, x):
    return self.block(x)


class Classifier_CNN(nn.Module):

    def __init__(self, input_shape, nb_classes):
        super(Classifier_CNN, self).__init__()
        self.nb_classes = nb_classes

        self.conv1 = conv_block(in_channels=1, out_channels=6, kernel_size=7,  stride=1, padding='valid')
        self.conv2 = conv_block(in_channels=6, out_channels=12, kernel_size=7,  stride=1, padding='valid')
        val = int(((input_shape[1] - 7) / 3) - 6)/3
        self.fc1 = nn.Linear(12 * int(val) , nb_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.sigmoid(x)

        return x


