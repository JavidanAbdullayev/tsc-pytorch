
import torch
from torch import nn


class mlp_block(nn.Module):

  def __init__(self, in_channels, out_channels, dropout):
    super(mlp_block, self).__init__()

    self.block = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_channels, out_channels),
        nn.ReLU()
    )


  def forward(self, x):
    return self.block(x)


class Classifier_MLP(nn.Module):

    def __init__(self, input_shape, nb_classes):
        super(Classifier_MLP, self).__init__()
        self.nb_classes = nb_classes
        self.in_channels = 10
        self.dense_1 = mlp_block(input_shape[-1], 500, 0.1)
        self.dense_2 = mlp_block(500, 500, 0.2)
        self.dense_3 = mlp_block(500, 500, 0.2)
        
        self.drop_1 = nn.Dropout(p=0.3)
        self.dense_4 = nn.Linear(500, nb_classes)



    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.drop_1(x)
        x = self.dense_4(x)

        return x

