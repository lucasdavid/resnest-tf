# ResNeSt: Split-Attention Networks

This is an implementation of "ResNeSt: Split-Attention Networks" on Keras and Tensorflow using native Convolution groups
and weights.

The implementation is based on [Hyungjin Kim's TF implementation](https://github.com/Burf/ResNeSt-Tensorflow2),
which in turn references on the [official Torch implementation](https://github.com/zhanghang1989/ResNeSt).

This implementation requires Tensorflow 2.9, as `Conv2D#groups` are used.

### Pretrained Weights

Weights were ported from the Torch implementation using the conversion scripts in [tools](/tools).
All networks were pre-trained over imagenet.

### Usage
```py
import tensorflow as tf
from resnest import ResNeSt101

# For classification (dilation=1):
model = resnest.ResNeSt101(
  input_shape=[512, 512, 3],
  weights='imagenet'
)

# For segmentation (dilation in (2, 4)):
model = resnest.ResNeSt101(
  input_shape=[512, 512, 3],
  weights='imagenet',
  include_top=False,
  pooling=None,
  dilated=4
)
```

#### Preprocessing
Data must be preprocessed with `tf.keras.applications.imagenet_utils.preprocess_input(x, mode='torch')`.
In other words:
```py
from keras.applications.imagenet_utils import preprocess_input

x = load_data()
x = preprocess_input(x, mode='torch')

# Or...
x /= 255
x -= tf.convert_to_tensor([0.485, 0.456, 0.406])
x /= tf.convert_to_tensor([0.229, 0.224, 0.225])
```
