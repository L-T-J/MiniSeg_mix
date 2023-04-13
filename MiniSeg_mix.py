import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SwitchNorm2d, SN_CS_Parallel_Attention_block, SN_PostRes2d

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        output = self.act(self.bn(self.conv(input)))
        return output


class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DilatedParallelConvBlockD2, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, out_planes, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv1 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, dilation=1, groups=out_planes, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=2, dilation=2, groups=out_planes, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        output = self.conv0(input)
        d1 = self.conv1(output)
        d2 = self.conv2(output)
        output = d1 + d2
        output = self.bn(output)
        return output


class DilatedParallelConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(DilatedParallelConvBlock, self).__init__()
        assert out_planes % 4 == 0
        inter_planes = out_planes // 4
        self.conv1x1_down = nn.Conv2d(in_planes, inter_planes, 1, padding=0, groups=1, bias=False)
        self.conv1 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=1, dilation=1, groups=inter_planes, bias=False)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=2, dilation=2, groups=inter_planes, bias=False)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=4, dilation=4, groups=inter_planes, bias=False)
        self.conv4 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=8, dilation=8, groups=inter_planes, bias=False)
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1)
        self.conv1x1_fuse = nn.Conv2d(out_planes, out_planes, 1, padding=0, groups=4, bias=False)
        self.attention = nn.Conv2d(out_planes, 4, 1, padding=0, groups=4, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)


    def forward(self, input):
        output = self.conv1x1_down(input)
        d1 = self.conv1(output)
        d2 = self.conv2(output)
        d3 = self.conv3(output)
        d4 = self.conv4(output)
        p = self.pool(output)
        d1 = d1 + p
        d2 = d1 + d2
        d3 = d2 + d3
        d4 = d3 + d4
        att = torch.sigmoid(self.attention(torch.cat([d1, d2, d3, d4], 1)))
        d1 = d1 + d1 * att[:, 0].unsqueeze(1)
        d2 = d2 + d2 * att[:, 1].unsqueeze(1)
        d3 = d3 + d3 * att[:, 2].unsqueeze(1)
        d4 = d4 + d4 * att[:, 3].unsqueeze(1)
        output = self.conv1x1_fuse(torch.cat([d1, d2, d3, d4], 1))
        output = self.act(self.bn(output))

        return output


class DownsamplerBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(DownsamplerBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, out_planes, 1, stride=1, padding=0, groups=1, bias=False)
        self.conv1 = nn.Conv2d(out_planes, out_planes, 5, stride=stride, padding=2, groups=out_planes, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        output = self.conv1(self.conv0(input))
        output = self.act(self.bn(output))
        return output


def split(x):
    c = int(x.size()[1])
    c1 = round(c // 2)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2


class MiniSeg(nn.Module):
    def __init__(self, classes=2, P1=2, P2=3, P3=8, P4=6, bn_momentum=0.2, aux=True):
        super(MiniSeg, self).__init__()
        self.original_size = 768
        self.D1 = int(P1/2)
        self.D2 = int(P2/2)
        self.D3 = int(P3/2)
        self.D4 = int(P4/2)
        self.aux = aux

        self.preBlock = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.Conv2d(24, 24, kernel_size=3, padding=1),
            # SwitchNorm2d(24, momentum=bn_momentum),
            # nn.ReLU(inplace=True)
        )
        self.pred = nn.Sequential(
            # nn.Conv2d(24, 24, kernel_size=3, padding=1),
            # SwitchNorm2d(24, momentum=bn_momentum),
            # nn.ReLU(inplace=True),
            nn.Conv2d(24, 2, kernel_size=1),
            SwitchNorm2d(2, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.long1 = DownsamplerBlock(24, 32, stride=2)
        self.down1 = ConvBlock(24, 32, stride=2)
        self.level1 = nn.ModuleList()
        self.level1_long = nn.ModuleList()
        for i in range(0, P1):
            self.level1.append(ConvBlock(32, 32))
        for i in range(0, self.D1):
            self.level1_long.append(DownsamplerBlock(32, 32, stride=1))

        self.cat1 = nn.Sequential(
                    nn.Conv2d(64, 64, 1, stride=1, padding=0, groups=1, bias=False),
                    nn.BatchNorm2d(64))

        self.long2 = DownsamplerBlock(32, 64, stride=2)
        self.down2 = DilatedParallelConvBlock(32, 64, stride=2)
        self.level2 = nn.ModuleList()
        self.level2_long = nn.ModuleList()
        for i in range(0, P2):
            self.level2.append(DilatedParallelConvBlock(64, 64))
        for i in range(0, self.D2):
            self.level2_long.append(DownsamplerBlock(64, 64, stride=1))

        self.cat2 = nn.Sequential(
                    nn.Conv2d(128, 128, 1, stride=1, padding=0, groups=1, bias=False),
                    nn.BatchNorm2d(128))

        self.long3 = DownsamplerBlock(64, 128, stride=2)
        self.down3 = DilatedParallelConvBlock(64, 128, stride=2)
        self.level3 = nn.ModuleList()
        self.level3_long = nn.ModuleList()
        for i in range(0, P3):
            self.level3.append(DilatedParallelConvBlock(128, 128))
        for i in range(0, self.D3):
            self.level3_long.append(DownsamplerBlock(128, 128, stride=1))

        self.cat3 = nn.Sequential(
                    nn.Conv2d(256, 256, 1, stride=1, padding=0, groups=1, bias=False),
                    nn.BatchNorm2d(256))

        self.long4 = DownsamplerBlock(128, 256, stride=2)
        self.down4 = DilatedParallelConvBlock(128, 256, stride=2)
        self.level4 = nn.ModuleList()
        self.level4_long = nn.ModuleList()
        for i in range(0, P4):
            self.level4.append(DilatedParallelConvBlock(256, 256))
        for i in range(0, self.D4):
            self.level4_long.append(DownsamplerBlock(256, 256, stride=1))

        # self.up4_conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        # self.up4_bn4 = nn.BatchNorm2d(64)
        # self.up4_act = nn.PReLU(64)

        # self.up3_conv4 = DilatedParallelConvBlockD2(64, 32)
        # self.up3_conv3 = nn.Conv2d(32, 32, 1, stride=1, padding=0)
        # self.up3_bn3 = nn.BatchNorm2d(32)
        # self.up3_act = nn.PReLU(32)

        # self.up2_conv3 = DilatedParallelConvBlockD2(32, 24)
        # self.up2_conv2 = nn.Conv2d(24, 24, 1, stride=1, padding=0)
        # self.up2_bn2 = nn.BatchNorm2d(24)
        # self.up2_act = nn.PReLU(24)

        # self.up1_conv2 = DilatedParallelConvBlockD2(24, 8)
        # self.up1_conv1 = nn.Conv2d(8, 8, 1, stride=1, padding=0)
        # self.up1_bn1 = nn.BatchNorm2d(8)
        # self.up1_act = nn.PReLU(8)

        self.back1 = nn.Sequential(
            SN_PostRes2d(504, 128),
            # SN_PostRes2d(256, 128),
            SN_PostRes2d(128, 128),
        )
        self.back2 = nn.Sequential(
            SN_PostRes2d(312, 64),
            # PostRes2d(128, 64),
            SN_PostRes2d(64, 64),
        )
        self.back3 = nn.Sequential(
            SN_PostRes2d(216, 32),
            # PostRes2d(96, 32),
            SN_PostRes2d(32, 32),
        )

        self.back4 = nn.Sequential(
            SN_PostRes2d(192, 48),
            SN_PostRes2d(48, 24),
            SN_PostRes2d(24, 24),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            SwitchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            SwitchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            SwitchNorm2d(32, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3),
            SwitchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(size=(64, 64))
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            SwitchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(size=(128, 128))

        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3),
            SwitchNorm2d(32, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(size=(256, 256))
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(32, 24, kernel_size=3),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(size=(512, 512)),
        )

        # ================================================================================================
        # Fully Connected Layers
        # ================================================================================================
        self.pool_preblock_gate1 =nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_preblock_gate2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool_preblock_gate3 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.pool_out1_gate2 =nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_out1_gate3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool_out2_gate3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_out1_gate0 = nn.Sequential(
            nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2),
            SwitchNorm2d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out2_gate0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
            SwitchNorm2d(32, momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out2_gate1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            SwitchNorm2d(32, momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out3_gate0 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=8, stride=8),
            SwitchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out3_gate1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            SwitchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))
        self.up_out3_gate2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            SwitchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))
        
        self.Fint_num = [128, 64, 32, 24]
        self.att3_1 = SN_CS_Parallel_Attention_block(F_g=128, F_x=24, F_int=self.Fint_num[0])
        self.att3_2 = SN_CS_Parallel_Attention_block(F_g=128, F_x=32, F_int=self.Fint_num[0])
        self.att3_3 = SN_CS_Parallel_Attention_block(F_g=128, F_x=64, F_int=self.Fint_num[0])
        self.att3_4 = SN_CS_Parallel_Attention_block(F_g=128, F_x=128, F_int=self.Fint_num[0])

        self.att2_1 = SN_CS_Parallel_Attention_block(F_g=64, F_x=24, F_int=self.Fint_num[1])
        self.att2_2 = SN_CS_Parallel_Attention_block(F_g=64, F_x=32, F_int=self.Fint_num[1])
        self.att2_3 = SN_CS_Parallel_Attention_block(F_g=64, F_x=64, F_int=self.Fint_num[1])
        self.att2_4 = SN_CS_Parallel_Attention_block(F_g=64, F_x=64, F_int=self.Fint_num[1])

        self.att1_1 = SN_CS_Parallel_Attention_block(F_g=32, F_x=24, F_int=self.Fint_num[2])
        self.att1_2 = SN_CS_Parallel_Attention_block(F_g=32, F_x=32, F_int=self.Fint_num[2])
        self.att1_3 = SN_CS_Parallel_Attention_block(F_g=32, F_x=32, F_int=self.Fint_num[2])
        self.att1_4 = SN_CS_Parallel_Attention_block(F_g=32, F_x=64, F_int=self.Fint_num[2])

        self.att0_1 = SN_CS_Parallel_Attention_block(F_g=24, F_x=24, F_int=self.Fint_num[3])
        self.att0_2 = SN_CS_Parallel_Attention_block(F_g=24, F_x=24, F_int=self.Fint_num[3])
        self.att0_3 = SN_CS_Parallel_Attention_block(F_g=24, F_x=32, F_int=self.Fint_num[3])
        self.att0_4 = SN_CS_Parallel_Attention_block(F_g=24, F_x=64, F_int=self.Fint_num[3])
        if self.aux:
            self.pred4 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(64, classes, 1, stride=1, padding=0))
            self.pred3 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(32, classes, 1, stride=1, padding=0))
            self.pred2 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(24, classes, 1, stride=1, padding=0))
        self.pred1 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(8, classes, 1, stride=1, padding=0))

    def forward(self, input):
        preinput = self.preBlock(input) #24
        long1 = self.long1(preinput)
        output1 = self.down1(preinput)
        output1_add = output1 + long1
        for i, layer in enumerate(self.level1):
            if i < self.D1:
                output1 = layer(output1_add) + output1
                long1 = self.level1_long[i](output1_add) + long1
                output1_add = output1 + long1
            else:
                output1 = layer(output1_add) + output1
                output1_add = output1 + long1

        output1_cat = self.cat1(torch.cat([long1, output1], 1))
        output1_l, output1_r = split(output1_cat)

        long2 = self.long2(output1_l + long1)
        output2 = self.down2(output1_r + output1)
        output2_add = output2 + long2
        for i, layer in enumerate(self.level2):
            if i < self.D2:
                output2 = layer(output2_add) + output2
                long2 = self.level2_long[i](output2_add) + long2
                output2_add = output2 + long2
            else:
                output2 = layer(output2_add) + output2
                output2_add = output2 + long2

        output2_cat = self.cat2(torch.cat([long2, output2], 1))
        output2_l, output2_r = split(output2_cat)

        long3 = self.long3(output2_l + long2)
        output3 = self.down3(output2_r + output2)
        output3_add = output3 + long3
        for i, layer in enumerate(self.level3):
            if i < self.D3:
                output3 = layer(output3_add) + output3
                long3 = self.level3_long[i](output3_add) + long3
                output3_add = output3 + long3
            else:
                output3 = layer(output3_add) + output3
                output3_add = output3 + long3

        output3_cat = self.cat3(torch.cat([long3, output3], 1))
        output3_l, output3_r = split(output3_cat)

        long4 = self.long4(output3_l + long3)
        output4 = self.down4(output3_r + output3)
        output4_add = output4 + long4
        for i, layer in enumerate(self.level4):
            if i < self.D4:
                output4 = layer(output4_add) + output4
                long4 = self.level4_long[i](output4_add) + long4
                output4_add = output4 + long4
            else:
                output4 = layer(output4_add) + output4
                output4_add = output4 + long4

        # up4_conv4 = self.up4_bn4(self.up4_conv4(output4))
        # up4 = self.up4_act(up4_conv4)

        # up4 = F.interpolate(up4, output3.size()[2:], mode='bilinear', align_corners=False)
        # up3_conv4 = self.up3_conv4(up4)
        # up3_conv3 = self.up3_bn3(self.up3_conv3(output3))
        # up3 = self.up3_act(up3_conv4 + up3_conv3)

        # up3 = F.interpolate(up3, output2.size()[2:], mode='bilinear', align_corners=False)
        # up2_conv3 = self.up2_conv3(up3)
        # up2_conv2 = self.up2_bn2(self.up2_conv2(output2))
        # up2 = self.up2_act(up2_conv3 + up2_conv2)

        # up2 = F.interpolate(up2, output1.size()[2:], mode='bilinear', align_corners=False)
        # up1_conv2 = self.up1_conv2(up2)
        # up1_conv1 = self.up1_bn1(self.up1_conv1(output1))
        # up1 = self.up1_act(up1_conv2 + up1_conv1)

        # if self.aux:
        #     pred4 = F.interpolate(self.pred4(up4), input.size()[2:], mode='bilinear', align_corners=False)
        #     pred3 = F.interpolate(self.pred3(up3), input.size()[2:], mode='bilinear', align_corners=False)
        #     pred2 = F.interpolate(self.pred2(up2), input.size()[2:], mode='bilinear', align_corners=False)
        # pred1 = F.interpolate(self.pred1(up1), input.size()[2:], mode='bilinear', align_corners=False)

        # if self.aux:
        #     return (pred1, pred2, pred3, pred4, )
        # else:
        #     return (pred1, )

        # output layers
        self.out_conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=8, stride=8),
            # nn.Conv2d(in_channels=128,out_channels=1,kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=4),
            # nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            # nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1),
            nn.Sigmoid()
        )
        # ============================================================================
        # Fully Connected Modules
        # ============================================================================

        # ---------Gate0--------------
        out1_gate0 = self.up_out1_gate0(output1)  # 32
        out2_gate0 = self.up_out2_gate0(output2)  # 64
        out3_gate0 = self.up_out3_gate0(output3)  # 128
        # out4_gate0 = self.up_out4_gate0(output4)


        # ---------Gate1--------------
        preblock_gate1 = self.pool_preblock_gate1(preinput)  # 24
        out2_gate1 = self.up_out2_gate1(output2)  # 64
        out3_gate1 = self.up_out3_gate1(output3)  # 128


        # ---------Gate2--------------
        preblock_gate2 = self.pool_preblock_gate2(preinput)  # 24
        out1_gate2 = self.pool_out1_gate2(output1)  # 32
        out3_gate2 = self.up_out3_gate2(output3)  # 128
        # out4_gate2 = self.up_out4_gate2(output4)


        # ---------Gate3--------------
        preblock_gate3 = self.pool_preblock_gate3(preinput)  # 24
        out1_gate3 = self.pool_out1_gate3(output1)  # 32
        out2_gate3 = self.pool_out2_gate3(output2)  # 64
        # out4_gate3 = self.up_out4_gate3(output4)


        up4 = self.up4(output4)  # 128
        # up4 = self.up4(output4)
        up4 = nn.functional.interpolate(up4, size=(96, 96), mode='nearest')
        deconv4 = self.deconv4(output4)
        up4_sig = up4 + deconv4
        up4 = torch.cat((up4, deconv4), dim=1)
        preblock_gate3 = self.att3_1(g=up4_sig, x=preblock_gate3)
        out1_gate3 = self.att3_2(g=up4_sig, x=out1_gate3)
        out2_gate3 = self.att3_3(g=up4_sig, x=out2_gate3)
        output3 = self.att3_4(g=up4_sig, x=output3)
        # gate3_fusion = torch.cat((preblock_gate3, out1_gate3, out2_gate3, output3), dim=1)  # 24+32+64+128=248
        # gate3_back = self.gate3_back(gate3_fusion)  # 128
        gate3 = torch.cat((up4, preblock_gate3, out1_gate3, out2_gate3, output3), dim=1)  # 128 + 128 = 256
        # gate3 = self.em_att_unit3(comb3)
        comb3 = self.back1(gate3)


        up3 = self.up3(comb3)
        up3 = nn.functional.interpolate(up3, size=(192, 192), mode='nearest')
        deconv3 = self.deconv3(comb3)
        up3_sig = up3 + deconv3
        up3 = torch.cat((up3, deconv3), dim=1)
        preblock_gate2 = self.att2_1(g=up3_sig, x=preblock_gate2)
        out1_gate2 = self.att2_2(g=up3_sig, x=out1_gate2)
        output2 = self.att2_3(g=up3_sig, x=output2)
        out3_gate2 = self.att2_4(g=up3_sig, x=out3_gate2)
        # gate2_fusion = torch.cat((preblock_gate2, out1_gate2, output2, out3_gate2), dim=1)  # 24+32+64+64=184
        # gate2_back = self.gate2_back(gate2_fusion)  # 64
        gate2 = torch.cat((up3, preblock_gate2, out1_gate2, output2, out3_gate2), dim=1)  # 64 + 64 = 128
        # gate2 = self.em_att_unit2(comb2)
        comb2 = self.back2(gate2)


        up2 = self.up2(comb2)  # 32
        up2 = nn.functional.interpolate(up2, size=(384, 384), mode='nearest')
        deconv2 = self.deconv2(comb2)
        up2_sig = up2 + deconv2
        up2 = torch.cat((up2, deconv2), dim=1)
        preblock_gate1 = self.att1_1(g=up2_sig, x=preblock_gate1)
        output1 = self.att1_2(g=up2_sig, x=output1)
        out2_gate1 = self.att1_3(g=up2_sig, x=out2_gate1)
        out3_gate1 = self.att1_4(g=up2_sig, x=out3_gate1)
        # gate1_fusion = torch.cat((preblock_gate1, output1, out2_gate1, out3_gate1), dim=1)  # 24+32+32+64=152
        # gate1_back = self.gate1_back(gate1_fusion)  # 32
        gate1 = torch.cat((up2, preblock_gate1, output1, out2_gate1, out3_gate1), dim=1)  # 32 + 32 = 64
        # gate1 = self.em_att_unit1(comb1)
        comb1 = self.back3(gate1)  # 32

        up1 = self.up1(comb1)  # 24
        up1 = nn.functional.interpolate(up1, size=(768, 768), mode='nearest')
        deconv1 = self.deconv1(comb1)
        up1_sig = up1 + deconv1
        up1 = torch.cat((up1, deconv1), dim=1)
        preinput = self.att0_1(g=up1_sig, x=preinput)
        out1_gate0 = self.att0_2(g=up1_sig, x=out1_gate0)
        out2_gate0 = self.att0_3(g=up1_sig, x=out2_gate0)
        out3_gate0 = self.att0_4(g=up1_sig, x=out3_gate0)
        gate0 = torch.cat((up1, preinput, out1_gate0, out2_gate0, out3_gate0), dim=1)  # 24 + 24 = 48
        # gate0 = self.em_att_unit0(out)
        out = self.back4(gate0)  # 48
        # out = self.outBlock(out)  # 24
        out = self.pred(out)
        output3 = self.out_conv1(comb3)
        output2 = self.out_conv2(comb2)
        output1 = self.out_conv3(comb1)
        return out, output1, output2, output3  # 640, 80, 160, 320
        # return out
