import torch
import torch.nn as nn
import numpy as np
import os
from argparse import ArgumentParser
import yaml
from model import Model
import pdb
import onnxruntime

parser = ArgumentParser()
parser.add_argument('--config', default='config/default.yaml')
parser.add_argument('--weights', required=True)
parser.add_argument('--device', default='')
args = parser.parse_args()

## Load config
with open(args.config) as f:
    cfg = yaml.safe_load(f)

cfg['save_dir'] = os.path.abspath(cfg['save_dir'])
assert os.path.exists(cfg['save_dir']), 'save folder not found'

if args.device == '-1':
    device = torch.device('cpu')
else:
    if args.device != '':
        cfg['gpu'] = args.device
    device = torch.device('cuda:%d'%cfg['gpu'][0]) if torch.cuda.is_available() else torch.device('cpu')

# ## Load model
model = Model(version=cfg['version'], nc=cfg['nc'], max_boxes=cfg['max_boxes'], is_training=False)
model_weights = torch.load(args.weights, map_location='cpu')['state_dict']
print('Successfully load weights from ', args.weights)
for key in list(model_weights):
    model_weights[key.replace("model.", "")] = model_weights.pop(key)
model.load_state_dict(model_weights, strict=True)
model.eval()
model.fuse()
print('Load successfully from checkpoint: %s'%args.weights)

def get_latest_opset():
    # Return max supported ONNX opset by this version of torch
    return max(int(k[14:]) for k in vars(torch.onnx) if 'symbolic_opset' in k)  # opset

torch.onnx.export(
            model,  # --dynamic only compatible with cpu
            torch.zeros((1, 3, 640, 640)),
            os.path.join(args.weights.replace('.ckpt', '.onnx')) ,
            verbose=False,
            opset_version=get_latest_opset(),
            do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
            input_names=['images'],
            output_names=['detections'],
            dynamic_axes=None)

## Test export model
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
]

sess_options = onnxruntime.SessionOptions()
sess_options.enable_profiling = False
session = onnxruntime.InferenceSession(args.weights.replace('.ckpt', '.onnx'), providers=providers, sess_options=sess_options)
session.get_modelmeta()
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

#infer by onnx
output = session.run([], {input_name:np.random.random((1, 3, 640, 640)).astype('float32')})
output = np.array(output) 
print(output.shape)
