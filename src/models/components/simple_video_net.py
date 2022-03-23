import torch
import torch.nn as nn
import torch.nn.functional as F


# batch input (batch_size, timestep, channel, height, weight) eg. (batch_size, 29, 1, 88, 88)
class VideoConv(nn.Module):
    """VideoCNN
    input: (N, D, C, H, W), need to be reshaped to (N, C, D, H, W)
    """

    # init
    def __init__(self, in_channels: int = 1, kernel_size=3, stride=1, padding=1, dilation=0):
        super().__init__()
        # self.conv1 = nn.Conv3d(in_channels, 8, kernel_size=kernel_size, stride=(2, 1, 1), padding=padding, dilation=dilation)
        # self.relu = nn.ReLU()
        # self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))
        # self.conv2 = nn.Conv3d(8, 16, kernel_size, stride, padding, dilation=dilation)
        # self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # self.conv3 = nn.Conv3d(16, 32, kernel_size, stride, padding, dilation=1)
        # self.conv4 = nn.Conv3d(32, 64, kernel_size, stride, padding, dilation=1)
        # # reduce dimension
        # self.conv5 = nn.Conv3d(64, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0)
        # self.conv6 = nn.Conv3d(32, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0)      # torch.Size([4, 16, 14, 5, 5])
        # self.fc1 = nn.Linear(5 * 5 * 14 * 16, 500)
        # self.log_softmax = nn.LogSoftmax(dim=1)

        self.conv1 = nn.Conv3d(
            in_channels, 8, kernel_size, stride=(2, 1, 1), padding=padding, dilation=dilation
        )
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.fc1 = nn.Linear(2 * 15 * 11 * 11, 500)

    # forward
    def forward(self, x):
        x = x.transpose(1, 2)

        # x = self.pool1(self.relu(self.conv1(x)))
        # x = self.pool2(self.relu(self.conv2(x)))
        # x = self.pool2(self.relu(self.conv3(x)))
        # x = self.pool2(self.relu(self.conv4(x)))
        # x = self.conv6(self.conv5(x))
        # # flatten
        # x = x.view(x.size(0), -1)
        # x = self.log_softmax(self.fc1(x))
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
