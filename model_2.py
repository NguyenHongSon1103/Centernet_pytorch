import contextlib
import math
import torch
import torch.nn as nn
from modules import Conv, Bottleneck, SPPF, C2f, Concat, ImplicitA, ImplicitM, Decoder
import yaml

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
    for i, (f, n, m, args) in enumerate(d['backbone']):  # from, number, module, args
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
    def __init__(self, version='n') -> None:
        super().__init__()
        #back bone
        with open('v8_pretrained/yolov8.yaml') as f:
            d = yaml.safe_load(f)
        self.backbone, self.save = parse_model(d, 3, version)

    def forward(self, inp):
        assert inp.shape[1] == 3
        out_bb = {}
        x = inp

        for i in range(len(self.backbone)):
            # print(i)
            # if i not in [11, 14, 17, 20]:
            x = self.backbone[i](x)
            if i in [2, 4, 6]:
                out_bb[i] = x
        
        # print(out_bb[2].shape, out_bb[4].shape, out_bb[6].shape, out_bb[9].shape)
            # elif i == 11:
            #     x = self.backbone[i]((x, out_bb[6]))
            # elif i == 14:
            #     x = self.backbone[i]((x, out_bb[4]))
            # elif i == 17:
            #     x = self.backbone[i]((x, out_bb[12]))
            # elif i == 20:
            #     x = self.backbone[i]((x, out_bb[9]))

        return out_bb[2], out_bb[4], out_bb[6], x

class Neck(nn.Module):
    def __init__(self, ch) -> None:
        super().__init__()
        self.concat = Concat(1)
        c = 64 #c=ch[1]
        self.c2f_1 = Conv(ch[4], c, k=1, bias=True) #for layer21
        self.c2f_2 = Conv(ch[3], c, k=1, bias=True) #for layer18
        self.c2f_3 = Conv(ch[2], c, k=1, bias=True) #for layer15
        
        self.up = nn.Upsample(None, 2, 'bilinear')
        self.concat = Concat(1)
        
        self.ia = ImplicitA(c, std=0.05)
        self.ims = nn.ModuleList([ImplicitM(c, std=0.05) for i in range(3)])

    def forward(self, out_15, out_18, out_21):
        up1 = nn.Upsample(None, 8, 'bilinear')(out_21)
        up1 = self.c2f_1(up1)
        
        up2 = nn.Upsample(None, 4, 'bilinear')(out_18)
        up2 = self.c2f_2(up2)

        up3 = nn.Upsample(None, 2, 'bilinear')(out_15)
        up3 = self.c2f_3(up3)

        # x = self.ia(self.concat([self.ims[0](up1), self.ims[1](up2), self.ims[2](up3)]))
        x = self.ims[0](up1) + self.ims[1](up2) + self.ims[2](up3)
        x = self.ia(x)
        return x

class IHead(nn.Module):
    def __init__(self, ch, nc=20) -> None:
        super().__init__()
        c = 64 #c = ch[1]
        self.ia, self.im = ImplicitA(c), ImplicitM(c)
        self.conv1 = Conv(c, c*3, k=3, s=1, bias=True)
        self.conv2 = Conv(c*3, c*2, k=3, s=1, bias=True)
        self.conv3 = Conv(c*2, c, k=3, s=1, bias=True)

        self.hm_out = nn.Sequential(
            Conv(c, c, 3, 1, bias=True), self.ia,
            Conv(c, c, 3, 1, bias=True), self.im,
            nn.Conv2d(c, nc, 1, bias=True),
            nn.Sigmoid()
        )

        self.wh_out = nn.Sequential(
            Conv(c, c, 3, 1, bias=True), self.ia,
            Conv(c, c, 3, 1, bias=True), self.im,
            nn.Conv2d(c, 2, 1, bias=True),
            # nn.ReLU()
        )

        self.reg_out = nn.Sequential(
            Conv(c, c, 3, 1, bias=True), self.ia,
            Conv(c, c, 3, 1, bias=True), self.im,
            nn.Conv2d(c, 2, 1, bias=True),
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

        self.backbone = Backbone(version)

        self.neck = Neck(ch)
        self.head = IHead(ch, nc)
        
        self.is_training = is_training
        self.decoder = Decoder(max_boxes)        
        
        self._init_weights()

    def forward(self, inp):
        out15, out18, out21 = self.backbone(inp)
        x = self.neck(out15, out18, out21)
        out = self.head(x)
        if not self.is_training:
            detections = self.decoder(out)
            return detections
        return out
    
    def _init_weights(self):
        try: 
            v8_pretrained = 'v8_pretrained/yolov8%s.pt'%self.version
            self.backbone.load_state_dict(torch.load(v8_pretrained, map_location='cpu'), strict=False)
            print('Load successfully yolov8%s backbone weights !'%self.version)
        except:
            print('Cannot load yolov8%s backbone weights !'%self.version)
        #For all module inside head and neck
        for m in list(self.head.modules()) + list(self.neck.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        #For only last conv2d in self.hm_out modules            
        m = self.head.hm_out[-2]
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, -4.6)  

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