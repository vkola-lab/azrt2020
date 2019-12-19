import torch
import torch.nn as nn

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

if __name__ == "__main__":
    model = Vanila_CNN(10, 0.5).cuda()
    input = torch.Tensor(3, 1, 181, 217, 181).cuda()
    output = model(input)
    print(output.shape)
