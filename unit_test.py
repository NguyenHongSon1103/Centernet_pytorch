import torch
import numpy as np
import yaml
from tqdm import tqdm
# import torchinfo

def test_model():
    from model_2 import Model, Backbone
 
    '''
    n -> 2.9M, s-> 9.7M, m->22.8M
    '''
    # backbone = Model('n', nc=5)
    backbone = Backbone('n')
    # torchinfo.summary(backbone, input_size=(1, 3, 640, 640), depth=1)
    total = sum(dict((p.data_ptr(), p.numel()) for p in backbone.parameters()).values())
    print(total)
    data = np.random.random((2, 3, 640, 640)).astype('float32')
    data = torch.from_numpy(data)
    # res = backbone.test(data)
    res = backbone(data)
    # print(res.shape)
    print(res[0].shape, res[1].shape, res[2].shape, res[3].shape)

def totensor(arr):
    return torch.from_numpy(arr)

def test_loss():
    from loss import FocalLoss, RegLoss
    fl_loss = FocalLoss()
    l1_loss = RegLoss()

    hm_label = np.random.random((2, 2, 160, 160)).astype('float32')
    hm_pred  = np.random.random((2, 2, 160, 160)).astype('float32')
    hm_loss  = fl_loss(torch.from_numpy(hm_pred), torch.from_numpy(hm_label)) 

    indices  = np.random.randint(0, 100*100, size=(2, 100))
    wh_label = np.random.random((2, 100, 2)).astype('float32')*200
    # wh_pred  = np.random.random((2, 100, 2)).astype('float32')*200
    wh_pred  = np.random.random((2, 2, 160, 160)).astype('float32')*200
    wh_loss  = l1_loss(totensor(wh_pred), totensor(indices) > 0, totensor(indices), totensor(wh_label))

    print(hm_loss, wh_loss)

def test_generator():
    from dataset.generator import Generator
    from torch.utils.data import DataLoader
    from utils import save_batch

    with open('config/default.yaml') as f:
        cfg = yaml.safe_load(f)
    
    train_dataset = Generator(cfg, mode='train')
    val_dataset   = Generator(cfg, mode='val')
    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size=cfg['batch_size'], num_workers=0)
    val_loader    = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=0)

    # val_dataset.generate_coco_format('val_labels.json')
    for batch_idx, (items, target, im_paths) in tqdm(enumerate(train_loader), total=len(train_loader)):
        save_batch(im_paths, items, 640, cfg['save_dir'], str(batch_idx)+'.jpg')
        # batch = train_loader[i]
        # print(train_dataset.data[i])
        # x = train_dataset[i]
        # print(x[0].shape, x[1][1][:3])

        # print(d[0].shape)
        # print(d[1][0].shape, d[1][1].shape, d[1][2].shape, d[1][3].shape)
        # print(d[2])
        if batch_idx == 5: 
            assert False

if __name__ == '__main__':
    #test_generator()
    test_model()