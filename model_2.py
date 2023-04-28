import contextlib
import math
import torch
import torch.nn as nn
from modules import Conv, Bottleneck, SPPF, C2f, Concat, ImplicitA, ImplicitM, Decoder, ScaleFeatureSelection
import yaml
import os

def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch, version='n'):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary into a PyTorch model
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'act', 'scales'))
    depth, width = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple'))
    depth, width, max_channels = scales[version]

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone']+d['head'][:-1]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        # if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
        #          BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x):
        if m in (Conv, Bottleneck, SPPF, C2f, nn.ConvTranspose2d):    
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            # if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x):
            if m in (C2f,):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class Backbone(nn.Module):
    def __init__(self, version='n', load_pretrained=True) -> None:
        super().__init__()
        #back bone
        DIR = os.path.join(os.getcwd(), 'v8_pretrained')

        with open(os.path.join(DIR, 'yolov8.yaml')) as f:
            d = yaml.safe_load(f)
        self.backbone, self.save = parse_model(d, 3, version)
        if load_pretrained:
            try: 
                v8_pretrained = os.path.join(DIR, 'yolov8%s.pt'%version)
                self.backbone.load_state_dict(torch.load(v8_pretrained, map_location='cpu'), strict=False)
                print('Load successfully yolov8%s backbone weights !'%version)
            except:
                print('Cannot load yolov8%s backbone weights !'%version)

    def forward(self, inp):
        assert inp.shape[1] == 3
        out_bb = []
        x = inp

        for i in range(10):
            x = self.backbone[i](x)
            if i in [2, 4, 6, 9]:
                out_bb.append(x)

        return out_bb
            

class Neck(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256,
                 bias=False,
                attention_type='scale_spatial'):
        '''
        bias: Whether conv layers have bias or not.
        '''
        super().__init__()
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(inner_channels, inner_channels//4, 3, padding=1, bias=bias)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

        self.concat_attention = ScaleFeatureSelection(inner_channels, inner_channels//4, attention_type=attention_type)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, features):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4
        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        fuse = self.concat_attention(fuse, [p5, p4, p3, p2])
        return fuse

class IHead(nn.Module):
    def __init__(self, c, nc=20) -> None:
        super().__init__()
        self.ia, self.im = ImplicitA(c), ImplicitM(c)
        self.conv1 = Conv(c, c*3, k=3, s=1)
        self.conv2 = Conv(c*3, c*2, k=3, s=1)
        self.conv3 = Conv(c*2, c, k=3, s=1)

        self.hm_out = nn.Sequential(
            Conv(c, c, 3, 1), self.ia,
            Conv(c, c, 3, 1), self.im,
            nn.Conv2d(c, nc, 1, bias=True),
            nn.Sigmoid()
        )
        self.hm_out[-2].bias.data.fill_(-4.6)

        self.wh_out = nn.Sequential(
            Conv(c, c, 3, 1), self.ia,
            Conv(c, c, 3, 1), self.im,
            nn.Conv2d(c, 2, 1),
            # nn.ReLU()
        )

        self.reg_out = nn.Sequential(
            Conv(c, c, 3, 1), self.ia,
            Conv(c, c, 3, 1), self.im,
            nn.Conv2d(c, 2, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        
        out_hm = self.hm_out(x)
        out_wh = self.wh_out(x)
        out_reg = self.reg_out(x)

        return out_hm, out_wh, out_reg

class Model(nn.Module):
    def __init__(self, version='n', nc=80, max_boxes= 100, is_training=True):
        super().__init__()
        self.version = version
        scales = dict(
        # [depth, width, max_channels]
        n= [0.33, 0.25, 1024],  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
        s= [0.33, 0.50, 1024],  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
        m= [0.67, 0.75, 768],   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
        l= [1.00, 1.00, 512],   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
        x= [1.00, 1.25, 512],   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
        )
        gd, gw, max_channels = scales[version]
        ch = [make_divisible(c_*gw, 8) for c_ in [64, 128, 256, 512, max_channels]] #16, 32, 64, 128, 256
        # repeat = [max(round(i*gd), 1) for i in [3, 6]]

        self.backbone = Backbone(version, is_training)
        in_channels = ch[1:]
        inner_channels = 64
        
        self.neck = Neck(in_channels= in_channels,
                         inner_channels=inner_channels)
        self.head = IHead(inner_channels, nc)
        
        self.is_training = is_training
        self.decoder = Decoder(max_boxes)        
        
    def forward(self, inp):
        features = self.backbone(inp)
        fuse = self.neck(features)
        out = self.head(fuse)
        if not self.is_training:
            detections = self.decoder(out)
            return detections
        return out

    def fuse(self):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.
        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.modules():
                if isinstance(m, Conv) and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
            # self.info(verbose=verbose)

        return self
    
    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.
        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.
        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model