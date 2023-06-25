import torch
import numpy as np
import yaml
from tqdm import tqdm
# import torchinfo
from time import time

def test_model():
    from model import Model, Backbone
 
    '''
    n -> 3.0M, s-> 11.8M, m-> 28.4M
    '''
    backbone = Model('n', nc=2, is_training=False)
    device = torch.device(0)
    backbone.to(device)
    backbone.fuse()
    # backbone = Backbone('n')
    # torchinfo.summary(backbone, input_size=(1, 3, 640, 640), depth=1)
    total = sum(dict((p.data_ptr(), p.numel()) for p in backbone.parameters()).values())
    print(total)
    with torch.no_grad():
        data = np.random.random((1, 3, 640, 640)).astype('float32')
        data = torch.from_numpy(data).to(device)
        res = backbone(data)
        # print(res[0].shape, res[1].shape, res[2].shape)

    mean_time = 0
    for i in range(10):
        with torch.no_grad():
            data = np.random.random((1, 3, 640, 640)).astype('float32')
            data = torch.from_numpy(data).to(device)

            s = time()
            res = backbone(data)
            t = time()
            mean_time += t - s
            print("Forward time: ", t - s)
    
    print("Mean forward time: ", mean_time/10)
    print('Average FPS: ', (10/mean_time))

def totensor(arr):
    return torch.from_numpy(arr)

def test_loss():
    from loss import FocalLoss, RegLoss, BboxLoss, BalancedL1Loss

    def __init__(self):
        super().__init__()

    fl_loss = FocalLoss()
    # l1_loss = RegLoss()
    l1_loss = BalancedL1Loss()
    # iou_loss = BboxLoss()

    hm_label = np.random.random((2, 2, 160, 160)).astype('float32')
    hm_pred  = np.random.random((2, 2, 160, 160)).astype('float32')
    hm_loss  = fl_loss(torch.from_numpy(hm_pred), torch.from_numpy(hm_label)) 

    indices  = np.random.randint(0, 100*100, size=(2, 100))
    wh_label = np.random.random((2, 100, 2)).astype('float32')*200
    wh_pred  = np.random.random((2, 100, 2)).astype('float32')*200
    wh_pred  = np.random.random((2, 2, 160, 160)).astype('float32')*200
    wh_loss  = l1_loss(totensor(wh_pred), totensor(indices) > 0, totensor(indices), totensor(wh_label))
    
    # test_loss = iou_loss(160, totensor(wh_pred), totensor(indices) > 0, totensor(indices), totensor(wh_label))
    # print(test_loss)
    print(hm_loss, wh_loss)

def test_backward():
    from loss import Loss
    from model import Model, Backbone
    loss_fn = Loss()
    
    hm_label = np.random.random((2, 2, 160, 160)).astype('float32')
    hm_label = torch.from_numpy(hm_pred).to(device)

    indices  = np.random.randint(0, 100*100, size=(2, 100))
    wh_label = np.random.random((2, 100, 2)).astype('float32')*200
    wh_label = np.random.random((2, 100, 2)).astype('float32')*200
    
    backbone = Model('n', nc=2)
    device = torch.device(1)
    backbone.to(device)
    # backbone = Backbone('n')   
    with torch.no_grad():
        data = np.random.random((16, 3, 640, 640)).astype('float32')
        data = torch.from_numpy(data).to(device)
        res = backbone(data)
        print(res[0].shape, res[1].shape, res[2].shape)

    mean_ftime = 0
    mean_btime = 0
    for i in range(10):
        data = np.random.random((8, 3, 640, 640)).astype('float32')
        data = torch.from_numpy(data).to(device)
        
        s = time()
        out = backbone(data)
        t = time()
        loss, loss_dict = loss_fn(output, targets)
        loss.backward()
        t2 = time()

        mean_ftime += t - s
        mean_btime += t2 - t
        print("Forward time: ", t - s)
        print("Backward time: ", t2 - t)
    
    print("Mean forward time: ", mean_ftime/10)
    print("Mean backward time: ", mean_btime/10)
    

def test_generator():
    from dataset.generator import Generator
    from torch.utils.data import DataLoader
    from utils import save_batch

    with open('config/default.yaml') as f:
        cfg = yaml.safe_load(f)
    
    train_dataset = Generator(cfg, mode='train')
    val_dataset   = Generator(cfg, mode='val')
    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=8)
    val_loader    = DataLoader(val_dataset, batch_size=16, num_workers=8)

    # val_dataset.generate_coco_format('val_labels.json')
    mean_time = 0
    s = time()
    # for batch_idx in range(len(train_dataset)):
    #     x = train_dataset[batch_idx]
    for batch_idx, (imgs, targets, im_paths) in enumerate(train_loader):
        t = time()
        print('load batch time: ', t-s)
        mean_time += t-s 
        s = t
        # save_batch(im_paths, imgs.numpy(), [tg.numpy() for tg in targets], 640, cfg['save_dir'], str(batch_idx)+'.jpg')
        
        # batch = train_loader[i]
        # print(train_dataset.data[i])
        # x = train_dataset[i]
        # print(x[0].shape, x[1][1][:3])

        # print(d[0].shape)
        # print(d[1][0].shape, d[1][1].shape, d[1][2].shape, d[1][3].shape)
        # print(d[2])
        if batch_idx == 20: 
            break
    print("Avg time per batch: ", mean_time/20)

def test_semi_generator():
    from SSOD.dataset import Generator
    from torch.utils.data import DataLoader

    with open('config/polyps_set_partial.yaml') as f:
        cfg = yaml.safe_load(f)
    
    train_dataset = Generator(cfg, mode='train')
    val_dataset   = Generator(cfg, mode='val')
    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=8)
    val_loader    = DataLoader(val_dataset, batch_size=16, num_workers=8)

    # val_dataset.generate_coco_format('val_labels.json')
    mean_time = 0
    s = time()
    for batch_idx, (unsup_weak_images, unsup_strong_images, images, targets) in enumerate(train_loader):
        print(batch_idx, unsup_weak_images.shape, unsup_strong_images.shape, images.shape)
        t = time()
        print('load batch time: ', t-s)
        mean_time += t-s 
        s = t
        if batch_idx == 20: 
            break
    print("Avg time per batch: ", mean_time/20)

if __name__ == '__main__':
    # test_semi_generator()
    # test_generator()
    test_model() 
    # test_loss()