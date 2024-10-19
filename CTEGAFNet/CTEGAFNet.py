import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvNextv2 import convnextv2_base, convnextv2_large
from math import log

class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    
    
class Extend_Cascaded_Partial_Decoder(nn.Module):
    def __init__(self, channel):
        super(Extend_Cascaded_Partial_Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample2 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample3 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample4 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample5 = ConvBR(2 * channel, 2 * channel, 3, padding=1)
        self.conv_upsample6 = ConvBR(3 * channel, 3 * channel, 3, padding=1)

        self.conv_concat2 = ConvBR(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv_concat4 = ConvBR(4 * channel, 4 * channel, 3, padding=1)
        self.conv4 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv6 = ConvBR(4 * channel, 4 * channel, 3, padding=1)
        self.conv7 = nn.Conv2d(4 * channel, 1, 1)

        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, zt4, zt3, zt2, zt1):
        zt4_1 = zt4  # 48
        zt3_1 = self.conv_upsample1(self.upsample(zt4)) * zt3  #
        zt2_1 = self.conv_upsample2(self.upsample(zt3_1)) * self.conv_upsample3(self.upsample(zt3)) * zt2
        zt1_1 = self.conv_upsample3(self.upsample(zt2_1)) * self.conv_upsample4(self.upsample(zt2)) * zt1

        zt3_2 = torch.cat((zt3_1, self.conv_upsample4(self.upsample(zt4_1))), 1)
        zt3_2 = self.conv_concat2(zt3_2)
        zt2_2 = torch.cat((zt2_1, self.conv_upsample5(self.upsample(zt3_2))), 1)
        zt2_2 = self.conv_concat3(zt2_2)
        zt1_2 = torch.cat((zt1_1, self.conv_upsample6(self.upsample(zt2_2))), 1)
        zt1_2 = self.conv_concat4(zt1_2)

        pc = self.conv6(zt1_2)
        pc = self.conv7(pc)

        return pc
    
class Fusion(nn.Module):
    def __init__(self, in_channels, out__channels):
        super(Fusion, self).__init__()
        self.conv3_1 = ConvBR(in_channels // 4, out__channels // 4, 3, 1, 1)
        self.dconv5_1 = ConvBR(in_channels // 4, out__channels // 4, 3, 1, 2, dilation=2)
        self.dconv7_1 = ConvBR(in_channels // 4, out__channels // 4, 3, 1, 3, dilation=3)
        self.dconv9_1 = ConvBR(in_channels // 4, out__channels // 4, 3, 1, 4, dilation=4)
        self.conv1_2 = Conv1x1(in_channels, in_channels)
        self.conv3_3_2 = ConvBR(out__channels, out__channels, 3, 1, 1)

    def forward(self, fusion_feature):
        xc = torch.chunk(fusion_feature, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        fusion_feature = self.conv3_3_2(fusion_feature + xx)

        return fusion_feature
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重
    
# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重


# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图

class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce_1 = nn.Sequential(ConvBR(in_channel, out_channel, 3, padding=1))
        self.reduce_2 = nn.Sequential(ConvBR(out_channel, out_channel, 3, padding=1))

    def forward(self, x):
        x = self.reduce_1(x)
        return self.reduce_2(x)

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(128, 64)  # 256是resnet,128是Convnext
        self.reduce4 = Conv1x1(1024, 64)  # 2048是resnet,1024是Convnext
        self.block = nn.Sequential(
            ConvBR(64 + 64, 64, 3, 1, 1),
            ConvBR(64, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 1))
        
        # self.reduce2 = Conv1x1(256, 64)
        self.reduce3 = Conv1x1(1, 64)
        self.sig = nn.Sigmoid()

    def forward(self, x4, x1, pg):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        pg = F.interpolate(pg, size, mode='bilinear', align_corners=False)
        pg = self.reduce3(pg)
        pg = self.sig(pg)
        
        # print(pg.shape, x1.shape, x5.shape)
        x1 = x1 + x1 * (1 - pg)
        x4 = x4 + x4 * pg
        x1 = x1 * (1-x4) + x1
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out
    
class TEGA(nn.Module):
    def __init__(self, in_channels, out_channel=64):
        super(TEGA, self).__init__()

        self.conv1_1 = Conv1x1(64, in_channels)
        self.conv1_2 = Conv1x1(in_channels, out_channel)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.fusion = Fusion(in_channels, in_channels)
        self.cbam = CBAM(out_channel)

    def forward(self, xg, x, pred):
        xg = self.conv1_1(xg)
        residual = x
        xsize = x.size()[2:]

        # reverse attention
        if x.size()[2:] != pred.size()[2:]:
            pred = F.interpolate(pred, size=x.size()[2:], mode='bilinear', align_corners=False)

        background_att = 1 - pred
        background_x = x * background_att

        # boudary attention
        edge_pred = pred
        pred_feature = x * edge_pred

        # high-frequency feature
        edge_input = F.interpolate(xg, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)
        attention_map = self.attention(fusion_feature)
        
        fusion_feature = fusion_feature * attention_map
        fusion_feature = self.fusion(fusion_feature)

        out = fusion_feature + residual
        out = self.conv1_2(out)
        out = self.cbam(out)
        return out

class TextureEncoder(nn.Module):
    def __init__(self):
        super(TextureEncoder, self).__init__()
        self.conv1 = ConvBR(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBR(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBR(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = ConvBR(96, 64, kernel_size=3, stride=1, padding=1)
        self.conv_out = ConvBR(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        xg = self.conv2(feat)
        pg = self.conv_out(self.conv3(feat))
        return xg, pg
    
    
class CTEGAFNet(nn.Module):
    def __init__(self, arc='convnextv2_base'):
        super(CTEGAFNet, self).__init__()
        if arc == 'convnextv2_base':
            print('--> using convnextv2_base right now')
            self.convnextv2 = convnextv2_base(pretrained=True)
            in_channel_list = [128, 256, 512, 1024]
        elif arc == 'res2net':
            print('--> using res2net right now')
            self.res2net = res2net50_v1b_26w_4s(pretrained=True)
            in_channel_list = [256, 512, 1024, 2048]
        elif arc == 'Convnext_large':
            print('--> using Convnext-large right now')
            self.convnextv2 = convnextv2_large(pretrained=True)
            in_channel_list = [192, 384, 768, 1536]
        else:
            raise Exception("Invalid Architecture Symbol: {}".format(arc))
            
        self.edge = EAM()
        self.texture_encoder = TextureEncoder()  # 输出特征xg,pg
        self.ecpd = Extend_Cascaded_Partial_Decoder(64)

        self.dr1 = DimensionalReduction(in_channel=in_channel_list[0], out_channel=32)
        self.dr2 = DimensionalReduction(in_channel=in_channel_list[1], out_channel=64)
        self.dr3 = DimensionalReduction(in_channel=in_channel_list[2], out_channel=128)
        self.dr4 = DimensionalReduction(in_channel=in_channel_list[3], out_channel=256)

        self.tega1 = TEGA(32)
        self.tega2 = TEGA(64)
        self.tega3 = TEGA(128)
        self.tega4 = TEGA(256)
        
        self.predictor = nn.Conv2d(64, 1, 1)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        image_shape = x.size()[2:]
        out = self.convnextv2(x)

        x1 = out[0]  # 128*96*96
        x2 = out[1]  # 256*48*48
        x3 = out[2]  # 512*24*24
        x4 = out[3]  # 1024*12*12
                
        ## TEecd ##
        xg, pg = self.texture_encoder(x)
        edge = self.edge(x4, x1, pg)
        edge_att = torch.sigmoid(edge)
        
        xr1 = self.dr1(x1)
        xr2 = self.dr2(x2)
        xr3 = self.dr3(x3)
        xr4 = self.dr4(x4)

        zt1 = self.tega1(xg, xr1, edge_att)
        zt2 = self.tega2(xg, xr2, edge_att)
        zt3 = self.tega3(xg, xr3, edge_att)
        zt4 = self.tega4(xg, xr4, edge_att)

        pc = self.ecpd(zt4,zt3,zt2,zt1)
        
        zt4 = F.interpolate(zt4, size=zt1.size()[2:], mode='bilinear', align_corners=False)
        zt3 = F.interpolate(zt3, size=zt1.size()[2:], mode='bilinear', align_corners=False)
        zt2 = F.interpolate(zt2, size=zt1.size()[2:], mode='bilinear', align_corners=False)
        
        zt3 = zt3 + self.predictor(zt4)
        zt2 = zt2 + self.predictor(zt3)
        zt1 = zt1 + self.predictor(zt2)
        
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)
        
        map1 = self.predictor(F.interpolate(zt1, size=image_shape, mode='bilinear', align_corners=False))
        map2 = self.predictor(F.interpolate(zt2, size=image_shape, mode='bilinear', align_corners=False))
        map3 = self.predictor(F.interpolate(zt3, size=image_shape, mode='bilinear', align_corners=False))
        map4 = self.predictor(F.interpolate(zt4, size=image_shape, mode='bilinear', align_corners=False))
        
        return self.upsample2(pc)+map1, self.upsample(pg), map1, map2, map3, map4, oe


if __name__ == '__main__':
    net = CTEGAFNet(arc='convnextv2_base').eval()
    inputs = torch.randn(1, 3, 384, 384)
    out = net(inputs)
    print(out[0].shape)


    print('===============================================================')
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from thop import profile

    net = CTEGAFNet(arc='convnextv2_base').cuda()
    data = torch.randn(1, 3, 384, 384).cuda()
    flops, params = profile(net, (data,))
    print('flops: %.2f G, params: %.2f M' % (flops / (1024 * 1024 * 1024), params / (1024 * 1024)))
    y = net(data)
    # for i in y:
    # print(i.shape)
