import sys
from collections import OrderedDict

import tensorflow as tf
import torch
import os

import resnest

sys.setrecursionlimit(1500)

CONVERSIONS = [
  ('resnest50', resnest.ResNeSt50),
  ('resnest101', resnest.ResNeSt101),
  ('resnest200', resnest.ResNeSt200),
  ('resnest269', resnest.ResNeSt269),
]

OUTPUT_DIR = 'weights'


WEIGHTS = {
  'resnest50': 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest50-528c19ca.pth',
  'resnest101': 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest101-22405ba7.pth',
  'resnest200': 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest200-75117900.pth',
  'resnest269': 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest269-0cc87c48.pth'
}


def convert_and_load_weights(keras_model, torch_url):
    torch_weight = torch.hub.load_state_dict_from_url(torch_url, map_location="cpu", progress=True, check_hash=True)

    weight = OrderedDict()
    for k, v in dict(torch_weight).items():
        if k.split(".")[-1] in ["weight", "bias", "running_mean", "running_var"]:
            if ("downsample" in k or "conv" in k) and "weight" in k and v.ndim == 4:
                v = v.permute(2, 3, 1, 0)
            elif "fc.weight" in k:
                v = v.t()
            weight[k] = v.cpu().data.numpy()

    downsample = []
    keras_weight = []
    for i, (torch_name, torch_weight) in enumerate(weight.items()):
        if i < len(keras_model.weights):
            if "downsample" in torch_name:
                downsample.append(torch_weight)
                continue
            else:
                torch_weight = [torch_weight]
            keras_weight += torch_weight

    for w in keras_model.weights:
        if "downsample" in w.name:
            new_w = downsample.pop(0)
        else:
            new_w = keras_weight.pop(0)
        tf.keras.backend.set_value(w, new_w)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for network_name, network_build in CONVERSIONS:
  input_tensor = tf.keras.Input([None, None, 3], name='images')
  model = network_build(input_tensor=input_tensor, weights=None)

  convert_and_load_weights(model, WEIGHTS[network_name])
  model.save_weights(os.path.join(OUTPUT_DIR, f'{network_name}.h5'))

  model = tf.keras.Model(model.input, model.get_layer('avg_pool').input, name=network_name)
  model.save_weights(os.path.join(OUTPUT_DIR, f'{network_name}_no_top.h5'))
