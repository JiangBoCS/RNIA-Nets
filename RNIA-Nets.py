import torch
import kornia
import random
import torch.nn as nn
import torch.nn.functional as F


class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()

        self.avg_pool_channel_1 = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.ReLU(inplace=True)
        )

        self.avg_pool_channel_2 = nn.AdaptiveAvgPool2d(1)
        self.fc_2 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.ReLU(inplace=True)
        )

        self.avg_pool_channel_3 = nn.AdaptiveAvgPool2d(1)
        self.fc_3 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h_size, w_size = x.size()

        f_1 = self.avg_pool_channel_1(x).view(b, c)

        f_1 = self.fc_1(f_1).view(b, c, 1, 1)

        x_1 = f_1 * x

        f_2 = self.avg_pool_channel_2(x_1).view(b, c)

        f_2 = self.fc_1(f_2).view(b, c, 1, 1)

        x_2 = f_2 * x_1

        f_3 = self.avg_pool_channel_3(x_2).view(b, c)

        f_3 = self.fc_1(f_3).view(b, c, 1, 1)

        y = f_1 + f_2 + f_3
        # y.shape is [batch_size, channel, 1, 1]
        return y


class HALayer(nn.Module):
    def __init__(self, channel):
        super(HALayer, self).__init__()

        self.fc_Horizontal_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            nn.ReLU(inplace=True)
        )

        self.fc_Horizontal_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            nn.ReLU(inplace=True)
        )

        self.fc_Horizontal_3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            nn.ReLU(inplace=True)
        )

        self.fc_Horizontal_4 = nn.Sequential(
            nn.Conv2d(3 * channel, channel, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h_size, w_size = x.size()
        self.avg_pool_Horizontal = nn.AdaptiveAvgPool2d((h_size, 1))

        h_pool_1 = self.avg_pool_Horizontal(x)
        h_y_1 = self.fc_Horizontal_1(h_pool_1)
        x_1 = h_y_1 * x

        h_pool_2 = self.avg_pool_Horizontal(x_1)
        h_y_2 = self.fc_Horizontal_2(h_pool_2)
        x_2 = h_y_2 * x_1

        h_pool_3 = self.avg_pool_Horizontal(x_2)
        h_y_3 = self.fc_Horizontal_3(h_pool_3)
        x_3 = h_y_3 * x_2

        xx = torch.cat((torch.cat((x_1, x_2), 1), x_3), 1)

        h_pool_4 = self.avg_pool_Horizontal(xx)
        h_y_4 = self.fc_Horizontal_4(h_pool_4)
        h_y = h_y_4
        # h_y.shape is [batch_size, channel, h_size, 1]

        return h_y


class WALayer(nn.Module):
    def __init__(self, channel):
        super(WALayer, self).__init__()

        self.fc_Vertical_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.ReLU(inplace=True)
        )

        self.fc_Vertical_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.ReLU(inplace=True)
        )

        self.fc_Vertical_3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.ReLU(inplace=True)
        )

        self.fc_Vertical_4 = nn.Sequential(
            nn.Conv2d(3 * channel, channel, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h_size, w_size = x.size()
        self.avg_pool_Vertical = nn.AdaptiveAvgPool2d((1, w_size))

        v_pool_1 = self.avg_pool_Vertical(x)
        v_y_1 = self.fc_Vertical_1(v_pool_1)
        x_1 = v_y_1 * x

        v_pool_2 = self.avg_pool_Vertical(x_1)
        v_y_2 = self.fc_Vertical_2(v_pool_2)
        x_2 = v_y_2 * x_1

        v_pool_3 = self.avg_pool_Vertical(x_2)
        v_y_3 = self.fc_Vertical_3(v_pool_3)
        x_3 = v_y_3 * x_2

        xx = torch.cat((torch.cat((x_1, x_2), 1), x_3), 1)

        v_pool_4 = self.avg_pool_Vertical(xx)
        v_y_4 = self.fc_Vertical_4(v_pool_4)
        v_y = v_y_4

        # v_y.shape is [batch_size, channel, 1, w_size]
        return v_y


class PALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(PALayer, self).__init__()

        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, y, z):
        pa_out_x = self.pa(x)
        pa_out_y = self.pa(y)
        pa_out_z = self.pa(z)
        # pa_out.shape is [batch_size, 1, h_size, w_size]
        return pa_out_x * pa_out_y * pa_out_z


class FFC_Block(nn.Module):
    def __init__(self, channel, feedback_num=1):
        super(FFC_Block, self).__init__()

        self.feedback = feedback_num

        self.in_put = nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=1, padding=0, bias=True)

        self.CA = CALayer(channel)
        self.HA = HALayer(channel)
        self.WA = WALayer(channel)
        self.PA = PALayer(channel)

        self.out_put = nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=1, padding=0, bias=True)

    def forward(self, x):
        y = x
        for i in range(self.feedback):
            y = self.in_put(y)
            CA_y = self.CA(y)
            HA_y = self.HA(y)
            WA_y = self.WA(y)
            PA_y = self.PA(CA_y, HA_y, WA_y)
            y = self.out_put(PA_y * x)

        output = y
        # pa_out.shape is [batch_size, 1, h_size, w_size]
        return output#, CA_y, HA_y, WA_y, PA_y

class Kernel(nn.Module):
    def __init__(self):
        super(Kernel, self).__init__()
        self.feature = nn.Sequential(
                        nn.Conv2d(32, 16, 16, 1, 1),
                        nn.Flatten(),
                        nn.GELU(),
                        nn.Dropout(0.2),
                        nn.Linear(144, 25))

    def forward(self, feature, img):
        b = img.size(0)
        kernel_v = self.feature(get_patch(feature))
        out = []
        for i in range(b):
            out.append(kornia.filter2D(img[i, :, :, :].unsqueeze(0), kernel_v.view(b, 5, 5)[i, :, :].unsqueeze(0))) #
        out_feature = torch.cat(out, dim=0)
        return out_feature

def get_patch(img_in, patch_size=16):
    _, _, ih, iw = img_in.size()
    ip = patch_size
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    img_out = img_in[:, :, iy:iy + ip, ix:ix + ip]
    return img_out

class model(nn.Module):

    def __init__(self, inputs = 3, outputs = 3, num_f = 32):
        super(model, self).__init__()

        self.Conv_1 = nn.Conv2d(inputs, num_f, 3, stride=1, padding=1, dilation=1, bias=True)
        self.Conv_2 = nn.Conv2d(num_f, num_f, 3, stride=2, padding=1, dilation=1, bias=True)
        self.Conv_3 = nn.Conv2d(num_f, num_f, 3, stride=2, padding=1, dilation=1, bias=True)
        self.Conv_4 = nn.Conv2d(num_f, num_f, 3, stride=2, padding=1, dilation=1, bias=True)
        self.Conv_5 = nn.Conv2d(num_f, num_f, 3, stride=2, padding=1, dilation=1, bias=True)
        self.Conv_6 = nn.Conv2d(num_f, num_f, 3, stride=1, padding=1, dilation=1, bias=True)

        self.Dconv_1 = nn.ConvTranspose2d(num_f, num_f, 6, stride=2, padding=2, output_padding=0, bias=True)
        self.Dconv_2 = nn.ConvTranspose2d(num_f, num_f, 6, stride=2, padding=2, output_padding=0, bias=True)
        self.Dconv_3 = nn.ConvTranspose2d(num_f, num_f, 6, stride=2, padding=2, output_padding=0, bias=True)
        self.Dconv_4 = nn.ConvTranspose2d(num_f, num_f, 6, stride=2, padding=2, output_padding=0, bias=True)
        self.Dconv_5 = nn.ConvTranspose2d(num_f, outputs, 3, stride=1, padding=1, output_padding=0, bias=True)


        self.CCLayer_1 = FFC_Block(num_f)
        self.CCLayer_2 = FFC_Block(num_f)
        self.CCLayer_3 = FFC_Block(num_f)
        self.CCLayer_4 = FFC_Block(num_f)

        self.k = Kernel()

        self.out = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1, output_padding=0, bias=True)

    def forward(self, x):
        conv_1 = F.relu(self.Conv_1(x))

        conv_2 = F.relu(self.Conv_2(conv_1))

        conv_3 = F.relu(self.Conv_3(conv_2))

        conv_4 = F.relu(self.Conv_4(conv_3))

        conv_5 = F.relu(self.Conv_5(conv_4))

        conv_out = F.relu(self.Conv_6(conv_5))

        Dconv_1 = F.relu(self.Dconv_1(conv_out + conv_5))

        DF1 = self.CCLayer_1(Dconv_1)

        Dconv_2 = F.relu(self.Dconv_2(DF1 + conv_4))

        DF2 = self.CCLayer_2(Dconv_2)

        Dconv_3 = F.relu(self.Dconv_3(DF2 + conv_3))

        DF3 = self.CCLayer_3(Dconv_3)

        Dconv_4 = F.relu(self.Dconv_4(DF3 + conv_2))

        DF4 = self.CCLayer_4(Dconv_4)

        Dconv_5 = F.relu(self.Dconv_5(DF4 + conv_1))

        Out = self.k(conv_out, Dconv_5)

        return Out  # , Dconv_5, Fconv_5, CA_y, HA_y, WA_y, PA_y