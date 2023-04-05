import torch
import numpy as np

def test_model():
    import yaml
    from model import Model
    with open('yolov8n.yaml') as f:
        d = yaml.safe_load(f)

    backbone = Model('n', nc=2)
    data = np.random.random((2, 3, 640, 640)).astype('float32')
    data = torch.from_numpy(data)
    res = backbone(data)
    print(res[0].shape, res[1].shape, res[2].shape)

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

if __name__ == '__main__':
    test_loss()