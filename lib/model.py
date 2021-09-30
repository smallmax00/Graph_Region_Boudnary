import torch
import torch.nn as nn
import torch.nn.functional as F
from .Res2Net_v1b import res2net50_v1b_26w_4s
import numpy as np

def get_gaussian_kernel(k=3, mu=0, sigma=3, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D

def get_sobel_kernel(k=3):

    range = np.linspace(-(k // 2), k // 2, k)

    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D

class generate_edge1(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3):
        super(generate_edge1, self).__init__()



        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)

        self.gaussian_filter.weight.data.copy_(torch.from_numpy(gaussian_2D))

        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)

        self.sobel_filter_x.weight.data.copy_(torch.from_numpy(sobel_2D))


        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)

        self.sobel_filter_y.weight.data.copy_(torch.from_numpy(sobel_2D.T))


        for param in self.gaussian_filter.parameters():
            param.requires_grad = False

        for param in self.sobel_filter_x.parameters():
            param.requires_grad = False
        for param in self.sobel_filter_y.parameters():
            param.requires_grad = False



    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):

        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).cuda()
        grad_x = torch.zeros((B, 1, H, W)).cuda()
        grad_y = torch.zeros((B, 1, H, W)).cuda()

        for c in range(C):
            grad_x = grad_x + self.sobel_filter_x(img[:, c:c + 1])
            grad_y = grad_y + self.sobel_filter_y(img[:, c:c + 1])


        grad_x, grad_y = grad_x / C, grad_y / C

        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5

        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(grad_magnitude[:, c:c+1])
        return blurred

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class Aggregation_seg(nn.Module):
    def __init__(self, in_fea=[32, 32, 32], mid_fea=16, out_fea=2):
        super(Aggregation_seg, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)



    def forward(self, high, low1, low2, low3):
        _, _, h, w = high.size()
        up_low1 = self.conv4(self.conv1(low1))
        up_low2 = self.conv4(self.conv2(low2))
        up_low3 = self.conv4(self.conv3(low2))
        cat = torch.cat((up_low1, up_low2, up_low3), dim=1)
        output = self.conv5(cat)
        return output

class Aggregation_edge(nn.Module):
    def __init__(self, in_fea=[32, 32, 32], mid_fea=16, out_fea=1):
        super(Aggregation_edge, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)



    def forward(self, high, low1, low2, low3):
        _, _, h, w = high.size()
        up_low1 = self.conv4(self.conv1(low1))
        up_low2 = self.conv4(self.conv2(low2))
        up_low3 = self.conv4(self.conv3(low2))
        cat = torch.cat((up_low1, up_low2, up_low3), dim=1)
        output = self.conv5(cat)
        return output

class ODOC_seg_edge_gru_gcn(nn.Module):

    def __init__(self, channel=32):
        super(ODOC_seg_edge_gru_gcn, self).__init__()

        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self.rfb2_1 = RFB_modified(256, channel)
        self.rfb3_1 = RFB_modified(512, channel)
        self.rfb4_1 = RFB_modified(1024, channel)
        self.rfb5_1 = RFB_modified(2048, channel)


        self.atten_depth_channel_0 = ChannelAttention(32)
        self.atten_depth_channel_1 = ChannelAttention(32)
        self.atten_depth_channel_2 = ChannelAttention(32)
        self.atten_depth_channel_3 = ChannelAttention(32)
        self.atten_depth_channel_4 = ChannelAttention(32)

        self.atten_depth_spatial_0 = SpatialAttention()
        self.atten_depth_spatial_1 = SpatialAttention()
        self.atten_depth_spatial_2 = SpatialAttention()
        self.atten_depth_spatial_3 = SpatialAttention()
        self.atten_depth_spatial_4 = SpatialAttention()


        self.atten_depth_channel_0_0 = ChannelAttention(32)
        self.atten_depth_channel_1_1 = ChannelAttention(32)
        self.atten_depth_channel_2_2 = ChannelAttention(32)
        self.atten_depth_channel_3_3 = ChannelAttention(32)
        self.atten_depth_channel_4_4 = ChannelAttention(32)

        self.atten_depth_spatial_0_0 = SpatialAttention()
        self.atten_depth_spatial_1_1 = SpatialAttention()
        self.atten_depth_spatial_2_2 = SpatialAttention()
        self.atten_depth_spatial_3_3 = SpatialAttention()
        self.atten_depth_spatial_4_4 = SpatialAttention()





        self.mlp1 = nn.Conv2d(2 * channel, channel, 1)
        self.mlp2 = nn.Conv2d(2 * channel, channel, 1)

        self.mlp3 = nn.Conv2d(2 * channel, channel, 1)

        self.mlp4 = nn.Conv2d(2 * channel, channel, 1)


        self.mlp11 = nn.Conv2d(2 * channel, channel, 1)
        self.mlp22 = nn.Conv2d(2 * channel, channel, 1)
        self.mlp33 = nn.Conv2d(2 * channel, channel, 1)
        self.mlp44 = nn.Conv2d(2 * channel, channel, 1)





        self.agg_seg = Aggregation_seg()
        self.agg_edge = Aggregation_edge()


        self.g_edge = generate_edge1()
        self.o_edge = nn.Sequential(
            BasicConv2d(2, 1, 3, padding=1),
            BasicConv2d(1, 1, 3, padding=1)
        )
        self.conv1x1 = BasicConv2d(1, 1, 1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)


        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)


        x1_o = self.rfb2_1(x1)
        x2_o = self.rfb3_1(x2)
        x3_o = self.rfb4_1(x3)
        x4_o = self.rfb5_1(x4)



        temp = x1_o.mul(self.atten_depth_channel_1(x1_o))
        temp = temp.mul(self.atten_depth_spatial_1(temp))
        x1_a = x1_o + temp
        x1_a = F.interpolate(x1_a + temp, size=(32, 32), mode='bilinear', align_corners=True)

        temp = x2_o.mul(self.atten_depth_channel_2(x2_o))
        temp = temp.mul(self.atten_depth_spatial_2(temp))
        x2_a = x2_o + temp


        temp = x3_o.mul(self.atten_depth_channel_3(x3_o))
        temp = temp.mul(self.atten_depth_spatial_3(temp))
        x3_a = x3_o + temp
        x3_a = F.interpolate(x3_a + temp, size=(32, 32), mode='bilinear', align_corners=True)


        temp = x4_o.mul(self.atten_depth_channel_4(x4_o))
        temp = temp.mul(self.atten_depth_spatial_4(temp))
        x4_a = x4_o + temp
        x4_a = F.interpolate(x4_a + temp, size=(32, 32), mode='bilinear', align_corners=True)



        temp = x1_o.mul(self.atten_depth_channel_1(x1_o))
        temp = temp.mul(self.atten_depth_spatial_1(temp))
        x1_1 = x1_o + temp
        x1_1 = F.interpolate(x1_1 + temp, size=(32, 32), mode='bilinear', align_corners=True)

        temp = x2_o.mul(self.atten_depth_channel_2_2(x2_o))
        temp = temp.mul(self.atten_depth_spatial_2_2(temp))
        x2_2 = x2_o + temp


        temp = x3_o.mul(self.atten_depth_channel_3(x3_o))
        temp = temp.mul(self.atten_depth_spatial_3(temp))
        x3_3 = x3_o + temp
        x3_3 = F.interpolate(x3_3 + temp, size=(32, 32), mode='bilinear', align_corners=True)


        temp = x4_o.mul(self.atten_depth_channel_4(x4_o))
        temp = temp.mul(self.atten_depth_spatial_4(temp))
        x4_4 = x4_o + temp
        x4_4 = F.interpolate(x4_4 + temp, size=(32, 32), mode='bilinear', align_corners=True)




        def gcn_f3(x1_a, x2_a, x3_a, x4_a, x1_1, x2_2, x3_3, x4_4):
            # x1
            e_x1_x2 = self.mlp1(torch.cat((x2_a - x1_a, x1_a), dim=1))
            e_x1_x3 = self.mlp1(torch.cat((x3_a - x1_a, x1_a), dim=1))
            e_x1_x4 = self.mlp1(torch.cat((x4_a - x1_a, x1_a), dim=1))

            e_x1_x11 = self.mlp1(torch.cat((x1_1 - x1_a, x1_a), dim=1))
            e_x1_x22 = self.mlp1(torch.cat((x2_2 - x1_a, x1_a), dim=1))
            e_x1_x33 = self.mlp1(torch.cat((x3_3 - x1_a, x1_a), dim=1))
            e_x1_x44 = self.mlp1(torch.cat((x4_4 - x1_a, x1_a), dim=1))
            ee_x1 = F.relu(e_x1_x2) * x2_a + F.relu(e_x1_x3) * x3_a + F.relu(e_x1_x4) * x4_a + \
                    F.relu(e_x1_x11) * x1_1 + F.relu(e_x1_x22) * x2_2 \
                    + F.relu(e_x1_x33) * x3_3 + F.relu(e_x1_x44) * x4_4
            # x2
            e_x2_x1 = self.mlp2(torch.cat((x1_a - x2_a, x2_a), dim=1))
            e_x2_x3 = self.mlp2(torch.cat((x3_a - x2_a, x2_a), dim=1))
            e_x2_x4 = self.mlp2(torch.cat((x4_a - x2_a, x2_a), dim=1))
            e_x2_x11 = self.mlp2(torch.cat((x1_1 - x2_a, x2_a), dim=1))
            e_x2_x22 = self.mlp2(torch.cat((x2_2 - x2_a, x2_a), dim=1))
            e_x2_x33 = self.mlp2(torch.cat((x3_3 - x2_a, x2_a), dim=1))
            e_x2_x44 = self.mlp2(torch.cat((x4_4 - x2_a, x2_a), dim=1))
            ee_x2 = F.relu(e_x2_x1) * x1_a + F.relu(e_x2_x3) * x3_a + F.relu(e_x2_x4) * x4_a \
                    + F.relu(e_x2_x11) * x1_1 + F.relu(e_x2_x22) * x2_2\
                    + F.relu(e_x2_x33) * x3_3 + F.relu(e_x2_x44) * x4_4
            # x3
            e_x3_x1 = self.mlp3(torch.cat((x1_a - x3_a, x3_a), dim=1))
            e_x3_x2 = self.mlp3(torch.cat((x2_a - x3_a, x3_a), dim=1))
            e_x3_x4 = self.mlp3(torch.cat((x4_a - x3_a, x3_a), dim=1))
            e_x3_x11 = self.mlp3(torch.cat((x1_1 - x3_a, x3_a), dim=1))
            e_x3_x22 = self.mlp3(torch.cat((x2_2 - x3_a, x3_a), dim=1))
            e_x3_x33 = self.mlp3(torch.cat((x3_3 - x3_a, x3_a), dim=1))
            e_x3_x44 = self.mlp3(torch.cat((x4_4 - x3_a, x3_a), dim=1))
            ee_x3 = F.relu(e_x3_x1) * x1_a + F.relu(e_x3_x2) * x2_a + F.relu(e_x3_x4) * x4_a \
                    + F.relu(e_x3_x11) * x1_1 + F.relu(e_x3_x22) * x2_2 \
                    + F.relu(e_x3_x33) * x3_3 + F.relu(e_x3_x44) * x4_4
            # x4
            e_x4_x1 = self.mlp4(torch.cat((x1_a - x4_a, x4_a), dim=1))
            e_x4_x2 = self.mlp4(torch.cat((x2_a - x4_a, x4_a), dim=1))
            e_x4_x3 = self.mlp4(torch.cat((x3_a - x4_a, x4_a), dim=1))
            e_x4_x11 = self.mlp4(torch.cat((x1_1 - x4_a, x4_a), dim=1))
            e_x4_x22 = self.mlp4(torch.cat((x2_2 - x4_a, x4_a), dim=1))
            e_x4_x33 = self.mlp4(torch.cat((x3_3 - x4_a, x4_a), dim=1))
            e_x4_x44 = self.mlp4(torch.cat((x4_4 - x4_a, x4_a), dim=1))
            ee_x4 = F.relu(e_x4_x1) * x1_a + F.relu(e_x4_x2) * x2_a + F.relu(e_x4_x3) * x3_a \
                    + F.relu(e_x4_x11) * x1_1 + F.relu(e_x4_x22) * x2_2\
                    + F.relu(e_x4_x33) * x3_3 + F.relu(e_x4_x44) * x4_4
            # x11
            e_x11_x1 = self.mlp22(torch.cat((x1_a - x1_1, x1_1), dim=1))
            e_x11_x2 = self.mlp22(torch.cat((x2_a - x1_1, x1_1), dim=1))
            e_x11_x3 = self.mlp22(torch.cat((x3_a - x1_1, x1_1), dim=1))
            e_x11_x4 = self.mlp22(torch.cat((x4_a - x1_1, x1_1), dim=1))
            e_x11_x22 = self.mlp22(torch.cat((x2_2 - x1_1, x1_1), dim=1))
            e_x11_x33 = self.mlp22(torch.cat((x3_3 - x1_1, x1_1), dim=1))
            e_x11_x44 = self.mlp22(torch.cat((x4_4 - x1_1, x1_1), dim=1))
            ee_x11 = F.relu(e_x11_x1) * x1_a + F.relu(e_x11_x2) * x2_a + F.relu(e_x11_x3) * x3_a + F.relu(
                e_x11_x4) * x4_a \
                     + F.relu(e_x11_x22) * x2_2 + F.relu(e_x11_x33) * x3_3 + F.relu(e_x11_x44) * x4_4

            # x22
            e_x22_x1 = self.mlp22(torch.cat((x1_a - x2_2, x2_2), dim=1))
            e_x22_x2 = self.mlp22(torch.cat((x2_a - x2_2, x2_2), dim=1))
            e_x22_x3 = self.mlp22(torch.cat((x3_a - x2_2, x2_2), dim=1))
            e_x22_x4 = self.mlp22(torch.cat((x4_a - x2_2, x2_2), dim=1))
            e_x22_x11 = self.mlp22(torch.cat((x1_1 - x2_2, x2_2), dim=1))
            e_x22_x33 = self.mlp22(torch.cat((x3_3 - x2_2, x2_2), dim=1))
            e_x22_x44 = self.mlp22(torch.cat((x4_4 - x2_2, x2_2), dim=1))
            ee_x22 = F.relu(e_x22_x1) * x1_a + F.relu(e_x22_x2) * x2_a + F.relu(e_x22_x3) * x3_a + F.relu(e_x22_x4) * x4_a \
                     + F.relu(e_x22_x11) * x1_1 + F.relu(e_x22_x33) * x3_3 + F.relu(e_x22_x44) * x4_4
            # x33
            e_x33_x1 = self.mlp33(torch.cat((x1_a - x3_3, x3_3), dim=1))
            e_x33_x2 = self.mlp33(torch.cat((x2_a - x3_3, x3_3), dim=1))
            e_x33_x3 = self.mlp33(torch.cat((x3_a - x3_3, x3_3), dim=1))
            e_x33_x4 = self.mlp33(torch.cat((x4_a - x3_3, x3_3), dim=1))
            e_x33_x11 = self.mlp33(torch.cat((x1_1 - x3_3, x3_3), dim=1))
            e_x33_x22 = self.mlp33(torch.cat((x2_2 - x3_3, x3_3), dim=1))
            e_x33_x44 = self.mlp33(torch.cat((x4_4 - x3_3, x3_3), dim=1))
            ee_x33 = F.relu(e_x33_x1) * x1_a + F.relu(e_x33_x2) * x2_a + F.relu(e_x33_x3) * x3_a + F.relu(e_x33_x4) * x4_a\
                     + F.relu(e_x33_x11) * x1_1 + F.relu(e_x33_x22) * x2_2 + F.relu(e_x33_x44) * x4_4
            # x44
            e_x44_x1 = self.mlp44(torch.cat((x1_a - x4_4, x4_4), dim=1))
            e_x44_x2 = self.mlp44(torch.cat((x2_a - x4_4, x4_4), dim=1))
            e_x44_x3 = self.mlp44(torch.cat((x3_a - x4_4, x4_4), dim=1))
            e_x44_x4 = self.mlp44(torch.cat((x4_a - x4_4, x4_4), dim=1))
            e_x44_x11 = self.mlp44(torch.cat((x1_1 - x4_4, x4_4), dim=1))
            e_x44_x22 = self.mlp44(torch.cat((x2_2 - x4_4, x4_4), dim=1))
            e_x44_x33 = self.mlp44(torch.cat((x3_3 - x4_4, x4_4), dim=1))
            ee_x44 = F.relu(e_x44_x1) * x1_a + F.relu(e_x44_x2) * x2_a + F.relu(e_x44_x3) * x3_a + F.relu(e_x44_x4) * x4_a\
                     + F.relu(e_x44_x11) * x1_1 + F.relu(e_x44_x22) * x2_2 + F.relu(e_x44_x33) * x3_3

            x1_u1 = ee_x1 + x1_a
            x2_u1 = ee_x2 + x2_a
            x3_u1 = ee_x3 + x3_a
            x4_u1 = ee_x4 + x4_a
            x11_u1 = ee_x11 + x1_1
            x22_u1 = ee_x22 + x2_2
            x33_u1 = ee_x33 + x3_3
            x44_u1 = ee_x44 + x4_4
            return x1_u1, x2_u1, x3_u1, x4_u1, x11_u1, x22_u1, x33_u1, x44_u1


        for i in range(3):
            x1_a, x2_a, x3_a, x4_a, x1_1, x2_2, x3_3, x4_4 = gcn_f3(x1_a, x2_a, x3_a, x4_a, x1_1, x2_2, x3_3, x4_4)


        seg_2 = self.agg_seg(x1_a, x2_a, x3_a, x4_a)

        edge_2 = self.agg_edge(x1_1, x2_2, x3_3, x4_4)


        seg_2_out = torch.sigmoid(seg_2)
        edge_2_out = torch.sigmoid(edge_2)


        seg_com = (seg_2[:, 0, :, :] + seg_2[:, 1, :, :]).unsqueeze(1)
        g_edge = self.g_edge(seg_com)

        fuse_edge = torch.cat((edge_2, g_edge), dim=1)
        out_edge = self.o_edge(fuse_edge) + edge_2
        out_edge = torch.sigmoid(out_edge)


        return seg_2_out, edge_2_out, out_edge


