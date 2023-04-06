import contextlib
import math
import torch
import torch.nn as nn
from modules import Conv, Bottleneck, SPPF, C2f, Concat, ImplicitA, ImplicitM, IHead, Decoder
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
        elif m in (Detect,):
            args.append([ch[x] for x in f])
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

class Neck(nn.Module):
    def __init__(self, ch, repeat) -> None:
        super().__init__()
        self.concat = Concat(1)
        self.c2f_1 = C2f(ch[4], ch[0]) #for layer9
        self.c2f_2 = C2f(ch[3], ch[0]) #for layer12
        self.c2f_3 = C2f(ch[2], ch[0]) #for layer15
        
        self.ia = ImplicitA(ch[0]*3)
        self.ims = [ImplicitM(ch[0]) for i in range(3)]

    def forward(self, out_9, out_12, out_15):
        up1 = nn.Upsample(None, 8, 'bilinear')(out_9)
        up1 = self.c2f_1(up1)
        
        up2 = nn.Upsample(None, 4, 'bilinear')(out_12)
        up2 = self.c2f_2(up2)

        up3 = nn.Upsample(None, 2, 'bilinear')(out_15)
        up3 = self.c2f_3(up3)

        x = self.ia(self.concat([self.ims[0](up1), self.ims[1](up2), self.ims[2](up3)]))

        return x

class Model(nn.Module):
    def __init__(self, version='n', nc=80, max_boxes= 100, is_training=True):
        super().__init__()
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
        repeat = [max(round(i*gd), 1) for i in [3, 6]]

        #back bone
        with open('yolov8n.yaml') as f:
            d = yaml.safe_load(f)
        self.backbone, _ = parse_model(d, 3, version)

        #neck
        self.layer_10_13 = nn.Upsample(None, 2, 'nearest')
        self.layer_11_14_17_20 = Concat(1)
        self.layer_12 = C2f(ch[3]+ch[4] , ch[3]) 
        self.layer_15 = C2f(ch[2]+ch[3] , ch[2])
        
        self.neck = Neck(ch, repeat)
        self.head = IHead(ch, nc)
        
        self.is_training = is_training
        self.decoder = Decoder(max_boxes)
        
    def forward(self, inp):
        assert inp.shape[1] == 3
        out_bb = []
        x = inp

        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i in [4, 6, 9]:
                out_bb.append(x)
        
        x = self.layer_10_13(x)
        x = self.layer_11_14_17_20((x, out_bb[1]))
        out_12 = self.layer_12(x)
        x = self.layer_10_13(out_12)
        x = self.layer_11_14_17_20((x, out_bb[0]))
        out_15 = self.layer_15(x)
        out = self.neck(out_bb[-1], out_12, out_15)
        out = self.head(out)
        if not self.is_training:
            detections = self.decoder(out)
            return detections
        return out
