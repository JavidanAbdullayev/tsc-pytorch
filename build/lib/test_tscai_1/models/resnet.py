import torch
from torch import nn

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



