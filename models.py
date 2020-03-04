import torch
import torch.nn as nn
import copy

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, kernel, pooling, BN=True, relu_type='leaky'):
        super().__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type=='leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate) 
       
    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

# define the discriminator
class Vanila_CNN(nn.Module):
    def __init__(self, fil_num, drop_rate):
        super(Vanila_CNN, self).__init__()
        self.block1 = ConvLayer(1, fil_num, 0.1, (4, 1, 0), (2, 1, 0))
        self.block2 = ConvLayer(fil_num, 2*fil_num, 0.1, (4, 1, 0), (2, 2, 0))
        self.block3 = ConvLayer(2*fil_num, 4*fil_num, 0.1, (3, 1, 0), (2, 2, 0))
        self.block4 = ConvLayer(4*fil_num, 8*fil_num, 0.1, (3, 1, 0), (2, 2, 0))
        self.block5 = ConvLayer(8*fil_num, 16*fil_num, 0.1, (3, 1, 0), (2, 2, 0))
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(16*fil_num*9*11*9, 30),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(30, 2),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x


# define the discriminator
class Vanila_CNN_Lite(nn.Module):
    def __init__(self, fil_num, drop_rate):
        super(Vanila_CNN_Lite, self).__init__()
        self.block1 = ConvLayer(1, fil_num, 0.1, (7, 2, 0), (3, 2, 0))
        self.block2 = ConvLayer(fil_num, 2*fil_num, 0.1, (4, 1, 0), (2, 2, 0))
        self.block3 = ConvLayer(2*fil_num, 4*fil_num, 0.1, (3, 1, 0), (2, 2, 0))
        self.block4 = ConvLayer(4*fil_num, 8*fil_num, 0.1, (3, 1, 0), (2, 1, 0))
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(8*fil_num*6*8*6, 30),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(30, 2),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

# define the generator
class _netG(nn.Module):
    def __init__(self, num):
        super(_netG, self).__init__()
        self.conv1 = nn.Conv3d(1, 2*num, kernel_size=(5,5,5), stride=(1,1,1), padding=(2,2,2))
        self.conv2 = nn.Conv3d(2*num, num, kernel_size=(1,1,1), stride=(1,1,1), padding=0)
        self.conv3 = nn.Conv3d(num, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.relu = nn.ReLU()
        self.BN1 = nn.BatchNorm3d(2*num)
        self.BN2 = nn.BatchNorm3d(num)

    def forward(self, Input):
        Input1 = self.relu(self.BN1(self.conv1(Input)))
        Input2 = self.relu(self.BN2(self.conv2(Input1)))
        Input3 = self.conv3(Input2)
        return Input3

# define the discriminator
class _netD(nn.Module):
    def __init__(self, num):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            # 50 50 50
            nn.Conv3d(1, num, 4, 2, 0, bias=False),
            nn.BatchNorm3d(num),
            nn.LeakyReLU(),
            # 24 24 24
            nn.Conv3d(num, 2*num, 4, 2, 0, bias=False),
            nn.BatchNorm3d(2*num),
            nn.LeakyReLU(),
            # 11 11 11
            nn.Conv3d(2*num, 4*num, 4, 2, 1, bias=False),
            nn.BatchNorm3d(4*num),
            nn.LeakyReLU(),
            # 5 5 5
            nn.Conv3d(4*num, 8*num, 3, 2, 0, bias=False),
            nn.BatchNorm3d(8*num),
            nn.LeakyReLU(),
            # 2, 2, 2
            nn.Conv3d(8*num, 1, 2, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).squeeze()


class _FCN(nn.Module):
    def __init__(self, num, p):
        super(_FCN, self).__init__()
        self.features = nn.Sequential(
            # 47, 47, 47
            nn.Conv3d(1, num, 4, 1, 0, bias=False),
            nn.MaxPool3d(2, 1, 0),
            nn.BatchNorm3d(num),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            # 43, 43, 43
            nn.Conv3d(num, 2*num, 4, 1, 0, bias=False),
            nn.MaxPool3d(2, 2, 0),
            nn.BatchNorm3d(2*num),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            # 20, 20, 20
            nn.Conv3d(2*num, 4*num, 3, 1, 0, bias=False),
            nn.MaxPool3d(2, 2, 0),
            nn.BatchNorm3d(4*num),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            # 9, 9, 9
            nn.Conv3d(4*num, 8*num, 3, 1, 0, bias=False),
            nn.MaxPool3d(2, 1, 0),
            nn.BatchNorm3d(8*num),
            nn.LeakyReLU(),
            # 6, 6, 6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(8*num*6*6*6, 30),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(30, 2),
        )
        self.feature_length = 8*num*6*6*6
        self.num = num

    def forward(self, x, stage='train'):
        x = self.features(x)
        if stage != 'inference':
            x = x.view(-1, self.feature_length)
        x = self.classifier(x)
        return x

    def dense_to_conv(self):
        fcn = copy.deepcopy(self)
        A = fcn.classifier[1].weight.view(30, 8*self.num, 6, 6, 6)
        B = fcn.classifier[4].weight.view(2, 30, 1, 1, 1)
        C = fcn.classifier[1].bias
        D = fcn.classifier[4].bias
        fcn.classifier[1] = nn.Conv3d(160, 30, 6, 1, 0).cuda()
        fcn.classifier[4] = nn.Conv3d(30, 2, 1, 1, 0).cuda()
        fcn.classifier[1].weight = nn.Parameter(A)
        fcn.classifier[4].weight = nn.Parameter(B)
        fcn.classifier[1].bias = nn.Parameter(C)
        fcn.classifier[4].bias = nn.Parameter(D)
        return fcn


if __name__ == "__main__":
    model = Vanila_CNN_Lite(10, 0.5).cuda()
    input = torch.Tensor(10, 1, 181, 217, 181).cuda()
    output = model(input)
    print(output.shape)
