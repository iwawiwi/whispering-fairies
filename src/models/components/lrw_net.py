# coding utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv2d_1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)

class BasicBlock(nn.Module):
    expansion = 1

    # init
    def __init__(
        self, 
        inplanes, 
        planes, 
        stride=1, 
        downsample=None, 
        se=False
    ):
        super().__init__()
        self.conv1 = conv2d_3x3(inplanes, planes, stride)
        self.conv2 = conv2d_3x3(planes, planes)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample            # downsample layer
        self.stride = stride
        self.se = se

        if self.se:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            # bottleneck layer
            self.conv3 = conv2d_1x1(planes, planes // 16)
            self.conv4 = conv2d_1x1(planes // 16, planes)
    
    # forward
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(self.bn(out))
        out = self.conv2(out)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.se:                         # squeeze and excitation
            w = self.global_pool(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()

            out = out * w                   # scaling output

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    # init
    def __init__(
        self, 
        block, 
        layers, 
        se=False
    ):
        super().__init__()
        self.inplanes = 64
        self.se = se
        
        self.layer1 = self.__make_layer(block, 64, layers[0])
        self.layer2 = self.__make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.__make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.__make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(512)                                       # batch normalization 1D

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))                 # init weight with normal distribution
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)                                      # init weight with 1
                m.bias.data.zero_()                                         # init bias with 0

    def __make_layer(self, block, planes, blocks, stride=1):
        """Make layer of blocks"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        return nn.Sequential(*layers)

    # forward pass
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)

        return x


class VideoCNN(nn.Module):
    # init
    def __init__(self, se=False):
        super().__init__()
        
        # 3d convolution frontend
        self.frontend3d = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        # resnet
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)
        #self.dropout = nn.Dropout(p=0.5)                                     # FIXME: dropout layer

        # initialize weights
        self.__init_weights()

    # frontend3d forward pass
    def forward_frontend3d(self, x):
        """Forward pass for 3d convolution frontend
        
        Args:
            x is a tensor of shape (batch_size, 29, 1, 88, 88)
        """
        x = x.transpose(1, 2)                       # swap channel with time dimension -> (batch_size, 1, 29, 88, 88)
        x = self.frontend3d(x)                      # output is (batch_size, 64, 29, 11, 11)
        x = x.transpose(1, 2)                       # swap time with channel dimension -> (batch_size, 29, 64, 11, 11)
        x = x.contiguous()                          # make x contiguous
        x = x.view(-1, 64, x.size(3), x.size(4))    # reshape to (batch_size * 29, 64, 11, 11)
        x = self.resnet18(x)                        # output is (batch_size * 29, 512)
        return x

    # forward pass
    def forward(self, x):
        """Forward pass for video CNN
        
        Args:
            x is a tensor of shape (batch_size, 29, 1, 88, 88)
        """
        b, t = x.size()[:2]             # batch size and time dimension
        x = self.forward_frontend3d(x)
        #x = self.dropout(x)
        x = x.view(b, -1, 512)          # reshape to (batch_size, 29, 512)
        return x

    # initialize weights as private method
    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))             # init weight with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()                                 # zero bias
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))             
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)                      # init weight with 1
                m.bias.data.zero_()                         # zero bias
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class LRWVideoModel(nn.Module):
    def __init__(
        self, 
        border=False, 
        se=False, 
        n_class=500, 
        dropout=0.5
    ):
        super().__init__()
        self.video_cnn = VideoCNN(se)
        self.border = border
        self.se = se
        self.n_class = n_class
        if border:
            in_dim = 512 + 1
        else:
            in_dim = 512
        # gru
        self.gru = nn.GRU(in_dim, 1024, num_layers=3, batch_first=True, bidirectional=True, dropout=0.2)    # TODO: why dropout=0.2?
        self.fc = nn.Linear(1024 * 2, n_class)
        self.dropout = nn.Dropout(p=dropout)

    # forward pass
    def forward(self, x, border=None):
        """Forward pass for LRWVideoModel
        
        Args:
            x is a tensor of shape (batch_size, 29, 1, 88, 88)
        """
        self.gru.flatten_parameters()       # TODO: why flatten parameters?

        # with autocast(), enabling mixed precision
        x = self.video_cnn(x)         # output is (batch_size, 29, 512)
        x = self.dropout(x)
        # convert to float32
        #x = x.float()

        if self.border:
            border = border[:,:,None]
            h, _ = self.gru(torch.cat((x, border), dim=2))   # output is (batch_size, 29, 1024 * 2)
        else:
            h, _ = self.gru(x)                               # output is (batch_size, 29, 1024 * 2)

        y = self.fc(self.dropout(h)).mean(dim=1)             # output is (batch_size, n_class)

        return y