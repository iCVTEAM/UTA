#!/usr/bin/python3
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Softmax, nn.Sigmoid, nn.ModuleList, nn.AdaptiveAvgPool2d)):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_bap=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample
        self.use_bap = use_bap

    def forward(self, x):
        # print(x.shape)
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out2 = F.relu(self.bn2(self.conv2(out1)), inplace=True)
        out3 = self.bn3(self.conv3(out2))
        if self.downsample is not None:
            x = self.downsample(x)
        if self.use_bap:
            return out1, out2, F.relu(out3 + x, inplace=True)
        else:
            return F.relu(out3 + x, inplace=True)
        # return F.relu(out3 + x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1, use_bap=False)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1, use_bap=False)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1, use_bap=False)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1, use_bap=False)

    def make_layer(self, planes, blocks, stride, dilation, use_bap=False):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        #layers.append(Bottleneck(self.inplanes, planes, dilation=dilation, use_bap=True))
        #if use_bap:
        #    return nn.Sequential(*layers)

        for _ in range(1, blocks-1):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))

        if use_bap:
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation, use_bap=True))
        else:
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        # out2_1, out2_2, out2 = self.layer1(out1)
        out2 = self.layer1(out1)
        # out3_1, out3_2, out3 = self.layer2(out2)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5_2 = self.layer4(out4)
        return out1, out2, out3, out4, out5_2

    def initialize(self):
        self.load_state_dict(torch.load('../../res/resnet50-19c8e357.pth'), strict=False)


class ASPP_246(nn.Module):

    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP_246, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()

        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

    def initialize(self):
        weight_init(self)


# cross-modality fusion: spatial perceptive module (SPM)
class CMF12(nn.Module):
    def __init__(self):
        super(CMF12, self).__init__()

        self.edge_conv = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(64),
        )
        self.fuse_conv = nn.Conv2d(128, 64, 3, 1, padding=1, bias=True)
        self.fuse_bn = nn.BatchNorm2d(64)

    def forward(self, sod, depth):
        if depth.size()[2:] != sod.size()[2:]:
            depth = F.interpolate(depth, size=sod.size()[2:], mode='bilinear', align_corners=True)

        # depth branch
        d_atten = nn.Sigmoid()(depth)

        # sod branch
        s_1 = sod * d_atten
        s_2 = sod + s_1

        # fuse branch
        f_1 = torch.cat((depth, s_2), dim=1)
        edge = self.edge_conv(f_1)
        edge_atten = nn.Sigmoid()(edge)

        f_2 = F.relu(self.fuse_bn(self.fuse_conv(f_1)), inplace=True)
        f_3 = f_2 * edge_atten
        f_4 = f_3 + f_2

        return f_4, edge

    def initialize(self):
        weight_init(self)


# channel-aware fusion module (CAF)
class CAF(nn.Module):
    def __init__(self):
        super(CAF, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v = nn.BatchNorm2d(64)

        self.convfuse = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnfuse = nn.BatchNorm2d(64)

        self.conv1f = nn.Conv2d(192, 192, kernel_size=1)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear', align_corners=True)
        out1h = F.relu(self.bn1h(self.conv1h(left)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down)), inplace=True)
        # mul = out1h * out1v
        mul = torch.mul(out1h, out1v)
        fuse = torch.cat((out1h, out1v), 1)
        fuse = torch.cat((fuse, mul), 1)

        gap = nn.AdaptiveAvgPool2d((1, 1))(fuse)
        # out1f = nn.Softmax(dim=1)(self.conv1f(gap)) * gap.shape[1]
        out1f = nn.Sigmoid()(self.conv1f(gap))
        # fuse_channel = out1f * fuse
        fuse_channel = torch.mul(out1f, fuse)

        out3h = F.relu(self.bn3h(self.conv3h(fuse_channel)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse_channel)), inplace=True)

        out4h = F.relu(self.bn4h(self.conv4h(out3h + out1h)), inplace=True)
        out4v = F.relu(self.bn4v(self.conv4v(out3v + out1v)), inplace=True)

        out = torch.cat((out4h, out4v), 1)
        out = F.relu(self.bnfuse(self.convfuse(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


class DASDecoder(nn.Module):
    def __init__(self):
        super(DASDecoder, self).__init__()
        # sod
        self.s45 = CAF()
        self.s34 = CAF()
        self.s23 = CAF()
        self.s12 = CAF()
        # cmf
        self.cmf5 = CMF12()
        self.cmf4 = CMF12()
        self.cmf3 = CMF12()
        self.cmf2 = CMF12()
        # depth
        self.d45 = CAF()
        self.d34 = CAF()
        self.d23 = CAF()

    def forward(self, out1s, out2s, out3s, out4s, out5s, out2d, out3d, out4d, out5d):
        # depth branch
        out4d = self.d45(out4d, out5d)
        out3d = self.d34(out3d, out4d)
        out2d = self.d23(out2d, out3d)

        # 5 after aspp
        cmf5s, edge5 = self.cmf5(out5s, out5d)
        # 4 after aspp
        out4s = self.s45(out4s, cmf5s)

        cmf4s, edge4 = self.cmf4(out4s, out4d)
        # 3
        out3s = self.s34(out3s, cmf4s)

        cmf3s, edge3 = self.cmf3(out3s, out3d)
        # 2
        out2s = self.s23(out2s, cmf3s)

        cmf2s, edge2 = self.cmf2(out2s, out2d)
        # # 1
        out1s = self.s12(out1s, cmf2s)

        return out1s, out2s, out3s, out4s, out5s, out2d, edge2, edge3, edge4, edge5

    def initialize(self):
        weight_init(self)


class UTA(nn.Module):
    def __init__(self, cfg):
        super(UTA, self).__init__()
        self.cfg = cfg
        self.bkbone = ResNet()
        # self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze_aspp = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        self.aspp = ASPP_246(512, 64)
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # depth branch
        self.squeeze_asppd = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        self.asppd = ASPP_246(512, 64)
        # self.squeeze5d = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4d = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3d = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2d = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.squeeze1d = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # self.decoder1 = SODDecoder()
        # self.depthdecoder = DepthDecoder()
        self.decoder = DASDecoder()
        self.linearr1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_88 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_80 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_72 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_64 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_56 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        # depth branch
        self.lineard2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineard3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineard4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineard5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        # edge branch
        self.lineare2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineare3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineare4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineare5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.initialize()

    def forward(self, x, shape=None):
        shape = x.size()[2:] if shape is None else shape

        out1h, out2h, out3h, out4h, out5v = self.bkbone(x)
        out5d = self.squeeze_asppd(out5v)
        out2d, out3d, out4d, out5d = self.squeeze2d(out2h), self.squeeze3d(out3h), self.squeeze4d(out4h), self.asppd(out5d)

        out5s = self.squeeze_aspp(out5v)
        out1h, out2h, out3h, out4h, out5v = self.squeeze1(out1h), self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.aspp(out5s)
        pred1, out2h, out3h, out4h, out5v, out2d, edge2, edge3, edge4, edge5 = \
            self.decoder(out1h, out2h, out3h, out4h, out5v, out2d, out3d, out4d, out5d)

        dpred = F.interpolate(self.lineard2(out2d), size=shape, mode='bilinear', align_corners=True)
        pred2 = F.interpolate(self.linearr1(pred1), size=shape, mode='bilinear')

        # gated multi-scale (GMS) module
        if self.cfg.mode == 'train':
            if pred1.shape[2] == 88:
                pred1 = F.interpolate(self.linearr1_88(pred1), size=shape, mode='bilinear')
            if pred1.shape[2] == 80:
                pred1 = F.interpolate(self.linearr1_80(pred1), size=shape, mode='bilinear')
            if pred1.shape[2] == 72:
                pred1 = F.interpolate(self.linearr1_72(pred1), size=shape, mode='bilinear')
            if pred1.shape[2] == 64:
                pred1 = F.interpolate(self.linearr1_64(pred1), size=shape, mode='bilinear')
            if pred1.shape[2] == 56:
                pred1 = F.interpolate(self.linearr1_56(pred1), size=shape, mode='bilinear')
        else:
            pred_88 = F.interpolate(pred1, size=[88, 88], mode='bilinear')
            pred_88 = F.interpolate(self.linearr1_88(pred_88), size=shape, mode='bilinear')
            pred_80 = F.interpolate(pred1, size=[80, 80], mode='bilinear')
            pred_80 = F.interpolate(self.linearr1_80(pred_80), size=shape, mode='bilinear')
            pred_72 = F.interpolate(pred1, size=[72, 72], mode='bilinear')
            pred_72 = F.interpolate(self.linearr1_72(pred_72), size=shape, mode='bilinear')
            pred_64 = F.interpolate(pred1, size=[64, 64], mode='bilinear')
            pred_64 = F.interpolate(self.linearr1_64(pred_64), size=shape, mode='bilinear')
            pred_56 = F.interpolate(pred1, size=[56, 56], mode='bilinear')
            pred_56 = F.interpolate(self.linearr1_56(pred_56), size=shape, mode='bilinear')
            pred1 = 1*pred_88 + 0.25*pred_80 + 0.25*pred_72 + 0.25*pred_64 + 0.25*pred_56

        out2h = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear', align_corners=True)
        out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear', align_corners=True)
        out4h = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear', align_corners=True)
        out5h = F.interpolate(self.linearr5(out5v), size=shape, mode='bilinear', align_corners=True)
        edge2 = F.interpolate(self.lineare2(edge2), size=shape, mode='bilinear', align_corners=True)
        edge3 = F.interpolate(self.lineare3(edge3), size=shape, mode='bilinear', align_corners=True)
        edge4 = F.interpolate(self.lineare4(edge4), size=shape, mode='bilinear', align_corners=True)
        edge5 = F.interpolate(self.lineare5(edge5), size=shape, mode='bilinear', align_corners=True)

        return pred1, pred2, out2h, out3h, out4h, out5h, dpred, edge2, edge3, edge4, edge5

    def initialize(self):
        if self.cfg.snapshot:
            device = torch.device('cuda')
            self.load_state_dict(torch.load(self.cfg.snapshot, map_location=device))
        else:
            weight_init(self)
