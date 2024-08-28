import torch
from torch import nn

norm_layer = nn.InstanceNorm2d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ks):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ks, stride=1,
                              padding=(ks - 1) // 2)
        self.bn = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        x_max = x.max(dim=1, keepdim=True)[0]
        return torch.cat([x_mean, x_max], dim=1)

class AttentionGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.compress = ZPool()
        self.conv = BasicConv2d(2, 1, kernel_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        y = self.compress(x)
        y = self.conv(y)
        y = self.activation(y)
        return x * y
    
class TripletAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.ch = AttentionGate(kernel_size)
        self.cw = AttentionGate(kernel_size)
        self.hw = AttentionGate(kernel_size)

    def forward(self, x):
        b, c, h, w = x.shape
        x_ch = self.ch(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # c and h
        x_cw = self.cw(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_hw = self.hw(x)
        return 1 / 3 * (x_ch + x_cw + x_hw)
    



class SCTA_main(nn.Module):
    def __init__(self, inplanes, planes,downsample=1, groups=32, ratio=16):
        # super(TripletAttention, self).__init__()
        super().__init__()
        d = max(planes // ratio, 32)
        self.planes = planes
        self.downsample=downsample
        self.attn3x3 = TripletAttention(kernel_size=3)#.to(device)
        self.attn5x5 = TripletAttention(kernel_size=5)#.to(device)
        
        self.split_3x3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,stride=self.downsample, groups=groups),
            norm_layer(planes),
            nn.ReLU()
        )
        self.split_5x5 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=2, dilation=2,stride=self.downsample, groups=groups),
            # nn.Conv2d(inplanes, planes, kernel_size=5, padding=2, stride=self.downsample, groups=groups),
            norm_layer(planes),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(planes, d),
            nn.InstanceNorm1d(d),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(d, planes)
        self.fc2 = nn.Linear(d, planes)
    

    def forward(self, x):
        batch_size = x.shape[0]
        # print(x.shape)
        u1 = self.attn3x3(self.split_3x3(x)) #self.split_3x3(x)#
        u2 = self.attn5x5(self.split_5x5(x))#self.split_5x5(x)#
        u = u1 + u2
        s = self.avgpool(u).flatten(1)
        z = self.fc(s)
        attn_scores = torch.cat([self.fc1(z), self.fc2(z)], dim=1)
        attn_scores = attn_scores.view(batch_size, 2, self.planes)
        attn_scores = attn_scores.softmax(dim=1)
        a = attn_scores[:,0].view(batch_size, self.planes, 1, 1)
        b = attn_scores[:,1].view(batch_size, self.planes, 1, 1)
        u1 = u1 * a.expand_as(u1)
        u2 = u2 * b.expand_as(u2)
        x = u1 + u2
        # print(x.shape)
        return x