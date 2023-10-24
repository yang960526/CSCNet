import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import functools
from einops import rearrange
import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from models import do_conv_pytorch as doconv
from .resnet import resnet34
import torch.nn.functional as F
##################bone###########
import matplotlib.pyplot as plt
##########################################我的网络都在这呢###############！！！！！！！！！！！！！！！！！！



###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'CSCNet':
        net = LHDACT(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'CDNet':
        net = CDNet()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    '''
    DO-Conv无痛涨点：使用over-parameterized卷积层提高CNN性能
    对于输入特征，先使用权重进行depthwise卷积，对输出结果进行权重为的传统卷积，
    '''
    return nn.Sequential(doconv.DOConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h

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

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = convbn(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              pad=padding, dilation=dilation)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class CSCM_1(nn.Module):
    def __init__(self, cur_channel):
        super(CSCM_1, self).__init__()
        self.relu = nn.ReLU(True)
        # current conv
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)
        self.cur_all = BasicConv2d(4*cur_channel, cur_channel, 3, padding=1)
        self.cur_all_ca = CoordAttention(cur_channel, cur_channel)

        # latter conv
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lat_sa = SpatialAttention()

    def forward(self, x_cur, x_lat):
        # current conv
        x_cur_1 = self.cur_b1(x_cur)
        x_cur_2 = self.cur_b2(x_cur)
        x_cur_3 = self.cur_b3(x_cur)
        x_cur_4 = self.cur_b4(x_cur)
        x_cur_all = self.cur_all(torch.cat((x_cur_1, x_cur_2, x_cur_3, x_cur_4), 1))
        cur_all_ca = self.cur_all_ca(x_cur_all)

        # latter conv
        x_lat = self.upsample2(x_lat)
        lat_sa = x_cur_all.mul(self.lat_sa(x_lat))

        x_LocAndGlo = cur_all_ca + lat_sa + x_cur

        return x_LocAndGlo


class CSCM_3(nn.Module):
    def __init__(self, cur_channel):
        super(CSCM_3, self).__init__()
        self.relu = nn.ReLU(True)

        # current conv
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)
        self.cur_all = BasicConv2d(4*cur_channel, cur_channel, 3, padding=1)
        self.cur_all_ca = CoordAttention(cur_channel, cur_channel)
        # previous conv
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()

    def forward(self, x_pre, x_cur):
        # current conv
        x_cur_1 = self.cur_b1(x_cur)
        x_cur_2 = self.cur_b2(x_cur)
        x_cur_3 = self.cur_b3(x_cur)
        x_cur_4 = self.cur_b4(x_cur)
        x_cur_all = self.cur_all(torch.cat((x_cur_1, x_cur_2, x_cur_3, x_cur_4), 1))
        cur_all_ca = self.cur_all_ca(x_cur_all)

        # previois conv
        x_pre = self.downsample2(x_pre)
        pre_sa = x_cur_all.mul(self.pre_sa(x_pre))

        x_LocAndGlo = cur_all_ca + pre_sa + x_cur

        return x_LocAndGlo

# for conv2
class CSCM_2(nn.Module):
    def __init__(self, cur_channel):
        super(CSCM_2, self).__init__()
        self.relu = nn.ReLU(True)

        # current conv
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)
        self.cur_all = BasicConv2d(4 * cur_channel, cur_channel, 3, padding=1)
        self.cur_all_ca = CoordAttention(cur_channel, cur_channel)
        self.cur_all_sa = SpatialAttention()
        # previous conv
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()

        # latter conv
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lat_sa = SpatialAttention()

    def forward(self, x_pre, x_cur, x_lat):
        # current conv
        x_cur_1 = self.cur_b1(x_cur)
        x_cur_2 = self.cur_b2(x_cur)
        x_cur_3 = self.cur_b3(x_cur)
        x_cur_4 = self.cur_b4(x_cur)
        x_cur_all = self.cur_all(torch.cat((x_cur_1, x_cur_2, x_cur_3, x_cur_4), 1))
        cur_all_ca = self.cur_all_ca(x_cur_all)

        # previois conv
        x_pre = self.downsample2(x_pre)
        pre_sa = x_cur_all.mul(self.pre_sa(x_pre))

        # latter conv
        x_lat = self.upsample2(x_lat)
        lat_sa = x_cur_all.mul(self.lat_sa(x_lat))

        x_LocAndGlo = cur_all_ca + pre_sa + lat_sa + x_cur

        return x_LocAndGlo

class TR(nn.Module):
    def __init__(self, token_len=4, enc_depth=1, dec_depth=4, in_channel=64,
                 dim_head=64, decoder_dim_head=64, decoder_softmax=True):
        super(TR, self).__init__()
        self.token_len=token_len
        dim = in_channel
        mlp_dim = 2*dim
        self.conv_a = nn.Conv2d(in_channels=in_channel, out_channels=token_len, kernel_size=1,
                                padding=0, bias=False)
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len, in_channel))

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_transformer(self, x):
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def forward(self,x):
        token = self._forward_semantic_tokens(x)
        token_new = self._forward_transformer(token)
        x = self._forward_transformer_decoder(x, token_new)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class decoder_TR(nn.Module):
    def __init__(self, channel=512):
        super(decoder_TR, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder3 = nn.Sequential(
            TR(in_channel=256),
            nn.Dropout(0.5),
            TransBasicConv2d(256, 128, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.decoder2 = nn.Sequential(
            TR(in_channel=256),
            nn.Dropout(0.5),
            TransBasicConv2d(256, 64, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.decoder1 = nn.Sequential(
            TR(in_channel=128)
        )

    def forward(self, x3, x2, x1):
        #  x3: 512, 32, 32; x2: 256, 64, 64; x1: 128, 128, 128
        x3_up = self.decoder3(x3)
        x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
        x1_up = self.decoder1(torch.cat((x1, x2_up), 1))
        return x1_up

class decoder_conv(nn.Module):
    def __init__(self, channel=512):
        super(decoder_conv, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder3 = nn.Sequential(
            BasicConv2d(256, 256, 3, padding=1, dilation=1),
            nn.Dropout(0.5),
            TransBasicConv2d(256, 128, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.decoder2 = nn.Sequential(
            BasicConv2d(256, 256, 3, padding=1, dilation=1),
            nn.Dropout(0.5),
            TransBasicConv2d(256, 64, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.decoder1 = nn.Sequential(
            BasicConv2d(128, 128, 3, padding=1, dilation=1),
        )

    def forward(self, x3, x2, x1):
        #  x3: 512, 32, 32; x2: 256, 64, 64; x1: 128, 128, 128
        x3_up = self.decoder3(x3)
        x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
        x1_up = self.decoder1(torch.cat((x1, x2_up), 1))
        return x1_up

class CSCNet(nn.Module):
    """
    resnet  + CSCM + TR_de + a small CNN
    """
    def __init__(self,):
        super(CSCNet, self).__init__()
        self.Resnet = resnet34(pretrained=True)
        self.cscm1 = CSCM_1(128)
        self.cscm2 = CSCM_2(256)
        self.cscm3 = CSCM_3(512)
        self.decoder = decoder_TR()
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.classifier = TwoLayerConv2d(in_channels=128, out_channels=2)
        self.conv1 = nn.Sequential(convbn(1024, 256, 3,1,1,1),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(convbn(512, 128, 3, 1, 1,1),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(convbn(256, 64, 3, 1, 1,1),
                                   nn.ReLU(True))

    def forward(self, x1, x2):

        outputs = []
        # forward backbone resnet
        X_A_1, X_A_2, X_A_3 = self.Resnet(x1)    # torch.Size([ 128, 128, 128])torch.Size([ 256, 64, 64])torch.Size([ 512, 32, 32])
        X_B_1, X_B_2, X_B_3 = self.Resnet(x2)

        X_A_1 = self.cscm1 (X_A_1, X_A_2)
        X_A_2 = self.cscm2(X_A_1, X_A_2, X_A_3)
        X_A_3 = self.cscm3(X_A_2, X_A_3)

        X_B_1 = self.cscm1(X_B_1, X_B_2)
        X_B_2 = self.cscm2(X_B_1, X_B_2, X_B_3)
        X_B_3 = self.cscm3(X_B_2, X_B_3)

        X_1 = torch.cat((X_A_1, X_B_1), dim=1)
        X_2 = torch.cat((X_A_2, X_B_2), dim=1)
        X_3 = torch.cat((X_A_3, X_B_3), dim=1)
        X_1 = self.conv3(X_1)
        X_2 = self.conv2(X_2)
        X_3 = self.conv1(X_3)

        X_1 = self.decoder(X_3, X_2, X_1)
        X = self.upsamplex2(X_1)
        X = self.classifier(X)
        outputs.append(X)

        return outputs

