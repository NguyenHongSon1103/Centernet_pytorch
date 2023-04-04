import contextlib
import math
import torch
import torch.nn as nn
from modules import Conv, Bottleneck, SPPF, C2f, Concat, Detect, ImplicitA, ImplicitM

def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch, version='n'):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary
    # nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']
    nc = d['nc']

    gd, gw = d['scales'][version][0], d['scales'][version][1]
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args 
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            # TODO: re-implement with eval() removal if possible
            # args[j] = (locals()[a] if a in locals() else ast.literal_eval(a)) if isinstance(a, str) else a
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
        #          BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x):
        if m in (Conv, Bottleneck, SPPF, C2f, nn.ConvTranspose2d):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            # if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x):
            if m in [C2f]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # elif m in (Detect, Segment):
        elif m in (Detect, ):
            args.append([ch[x] for x in f])
            # if m is Segment:
            #     args[2] = make_divisible(args[2] * gw, 8)
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

class Backbone(nn.Module):
    def __init__(self, version='n'):
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
        ch = [make_divisible(c_*gw, 8) for c_ in [64, 128, 256, 512, 1024]] #16, 32, 64, 128, 256
        repeat = [max(round(i*gd), 1) for i in [3, 6]]

        #back bone
        self.layer_0 = Conv(3, ch[0], k=3, s=2)
        self.layer_1 = Conv(ch[0], ch[1], k=3, s=2)
        self.layer_2 = nn.Sequential(*(C2f(ch[1], ch[1], shortcut=True) for _ in range(repeat[0])))
        self.layer_3 = Conv(ch[1], ch[2], k=3, s=2)
        self.layer_4 = nn.Sequential(*(C2f(ch[2], ch[2], shortcut=True) for _ in range(repeat[1])))
        self.layer_5 = Conv(ch[2], ch[3], k=3, s=2)
        self.layer_6 = nn.Sequential(*(C2f(ch[3], ch[3], shortcut=True) for _ in range(repeat[1])))
        self.layer_7 = Conv(ch[3], ch[4], k=3, s=2)
        self.layer_8 = nn.Sequential(*(C2f(ch[4], ch[4], shortcut=True) for _ in range(repeat[0])))
        self.layer_9 = SPPF(ch[4], ch[4])

        #neck
        self.layer_10_13 = nn.Upsample(None, 2, 'nearest')
        self.layer_11_14_17_20 = Concat(1)
        self.layer_12 = nn.Sequential(*(C2f(ch[3]+ch[4], ch[3]) for _ in range(repeat[0])))
        self.layer_15 = nn.Sequential(*(C2f(ch[2]+ch[3], ch[2]) for _ in range(repeat[0])))
        # original
        # self.layer_16 = Conv(ch[2], ch[2], k=3, s=2) ##Focus here
        # self.layer_18 = nn.Sequential(*(C2f(ch[2]+ch[3], ch[2]) for _ in range(repeat[0])))
        # self.layer_19 = Conv(ch[3], ch[3], k=3, s=2)
        # self.layer_21 = nn.Sequential(*(C2f(ch[3]+ch[4], ch[4]) for _ in range(repeat[0])))
        
    def forward(self, inp):
        assert inp.shape[1] == 3
        out_bb = []
        x = inp
        for i, l in enumerate([self.layer_0, self.layer_1, self.layer_2, self.layer_3, 
                            self.layer_4, self.layer_5, self.layer_6, self.layer_7,
                            self.layer_8, self.layer_9]):
            x = l(x)
            if i in [4, 6, 9]:
                out_bb.append(x)
        
        x = self.layer_10_13(x)
        x = self.layer_11_14_17_20((x, out_bb[1]))
        out_12 = self.layer_12(x)
        x = self.layer_10_13(out_12)
        x = self.layer_11_14_17_20((x, out_bb[0]))
        out_15 = self.layer_15(x)
        return out_12, out_15

class Neck(nn.Module):
    def __init__(self, ch, repeat) -> None:
        super().__init__()
        self.concat = Concat(1)
        self.c2f_1 = nn.Sequential(*(C2f(ch[4], ch[0]) for _ in range(repeat[0]))) #for layer9
        self.c2f_2 = nn.Sequential(*(C2f(ch[3], ch[0]) for _ in range(repeat[0]))) #for layer12
        self.c2f_3 = nn.Sequential(*(C2f(ch[2], ch[0]) for _ in range(repeat[0]))) #for layer15
        self.ia = ImplicitA(ch[0])
        self.ims = [ImplicitM(ch[0]) for i in range(3)]

    def forward(self, x):
        up1 = nn.Upsample(None, 8, 'bilinear')(x)
        up1 = self.c2f_1(up1)
        
        up2 = nn.Upsample(None, 4, 'bilinear')(x)
        up2 = self.c2f_2(up2)

        up3 = nn.Upsample(None, 2, 'bilinear')(x)
        up1 = self.c2f_3(up3)

        x = self.ia(self.concat([self.ims[0](up1), self.ims[1](up2), self.ims[2](up3)]))
        return x

         
if __name__ == '__main__':
    ## load yaml
    import numpy as np
    import yaml
    with open('yolov8n.yaml') as f:
        d = yaml.safe_load(f)
    # print(d)
    # backbone, _ = parse_model(d, 3)
    # for i in range(25):
    #     print('='*5, i, '='*5)
    #     print(backbone[i])
    # state_dict = torch.load('yolov8n.pt')
    # print(state_dict)
    backbone = Backbone('n')
    data = np.random.random((2, 3, 640, 640)).astype('float32')
    data = torch.from_numpy(data)
    res = backbone(data)
    print(res[0].shape, res[1].shape)