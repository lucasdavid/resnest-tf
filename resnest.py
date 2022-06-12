from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
from keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    MaxPool2D,
    Reshape,
)
from keras.utils import data_utils
from keras.utils import layer_utils
from keras.applications import imagenet_utils

CV_DEFAULTS = dict(padding="same", use_bias=False, kernel_initializer="he_normal")
BN_DEFAULTS = dict(
    axis=-1,
    momentum=0.9,
    epsilon=1e-5,
)

WEIGHTS_HASHES = {
    "resnest50": {
        "url": "https://github.com/lucasdavid/resnest-tf/releases/download/v0.0.1/resnest50.h5",
        "hash": "ebc1bb7cf7e960c9de7061e06c516fcb51885915ffede7869452c2180ec34759",
    },
    "resnest101": {
        "url": "https://github.com/lucasdavid/resnest-tf/releases/download/v0.0.1/resnest101.h5",
        "hash": "63b9202b4e6dc4019ea98cba649a36f6990afd194b0696d4a733729adffd017b",
    },
    "resnest200": {
        "url": "https://github.com/lucasdavid/resnest-tf/releases/download/v0.0.1/resnest200.h5",
        "hash": "9199062c60ff5d5f8248cd518321660c731af7e8b28f5ece6d86413aff63d94d",
    },
    "resnest269": {
        "url": "https://github.com/lucasdavid/resnest-tf/releases/download/v0.0.1/resnest269.h5",
        "hash": "e8453a58bfb15e98ab73492752aebfc3aa406ba5ed54e0f17da2eced48b579fe",
    },
}


@tf.keras.utils.register_keras_serializable(package="resnest")
class rSoftMax(tf.keras.layers.Layer):
    def __init__(self, filters, radix, group_size, **kwargs):
        super(rSoftMax, self).__init__(**kwargs)

        self.filters = filters
        self.radix = radix
        self.group_size = group_size

        if 1 < radix:
            self.seq1 = tf.keras.layers.Reshape(
                [group_size, radix, filters // group_size]
            )
            self.seq2 = tf.keras.layers.Permute([2, 1, 3])
            self.seq3 = tf.keras.layers.Softmax(axis=1)
            self.seq4 = tf.keras.layers.Reshape([1, 1, radix * filters])
            self.seq = [self.seq1, self.seq2, self.seq3, self.seq4]
        else:
            self.seq1 = Activation(tf.keras.activations.sigmoid)
            self.seq = [self.seq1]

    def call(self, inputs):
        out = inputs
        for l in self.seq:
            out = l(out)
        return out

    def get_config(self):
        config = super(rSoftMax, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "radix": self.radix,
                "group_size": self.group_size,
            }
        )

        return config


def splat_block(
    x: tf.Tensor,
    n_filter: int,
    kernel_size: int = 3,
    stride_size: int = 1,
    dilation: int = 1,
    group_size: int = 1,
    radix: int = 1,
    dropout_rate: float = 0.0,
    expansion: int = 4,
    prefix: str = "",
    cv_args: Optional[Dict[str, Union[str, int, float]]] = None,
    bn_args: Optional[Dict[str, Union[str, int, float]]] = None,
):
    if not cv_args:
        cv_args = CV_DEFAULTS
    if not bn_args:
        bn_args = BN_DEFAULTS

    if len(prefix) != 0:
        prefix += "_"

    y = Conv2D(
        n_filter * radix,
        kernel_size,
        strides=stride_size,
        dilation_rate=dilation,
        groups=group_size * radix,
        name=f"{prefix}splat_conv1",
        **cv_args,
    )(x)
    y = BatchNormalization(name=f"{prefix}splat_bn1", **bn_args)(y)
    if 0 < dropout_rate:
        y = Dropout(dropout_rate, name=f"{prefix}splat_dropout1")(y)
    y = Activation("relu", name=f"{prefix}splat_act1")(y)

    inter_channel = max(tf.keras.backend.int_shape(x)[-1] * radix // expansion, 32)
    if 1 < radix:
        split = tf.split(y, radix, axis=-1)
        y = Add(name=f"{prefix}splat_add")(split)
    y = GlobalAveragePooling2D(name=f"{prefix}splat_gap")(y)
    y = Reshape([1, 1, n_filter], name=f"{prefix}splat_expand_dims")(y)

    y = Conv2D(
        inter_channel,
        kernel_size=1,
        padding="same",
        groups=group_size,
        kernel_initializer="he_normal",
        name=f"{prefix}splat_conv2",
    )(y)
    y = BatchNormalization(name=f"{prefix}splat_bn2", **bn_args)(y)
    y = Activation("relu", name=f"{prefix}splat_act2")(y)
    y = Conv2D(
        n_filter * radix,
        kernel_size=1,
        groups=group_size,
        padding="same",
        use_bias=True,
        kernel_initializer="he_normal",
        name=f"{prefix}splat_conv3",
    )(y)

    a = rSoftMax(n_filter, radix, group_size, name=f"{prefix}splat_sm")(y)
    if 1 < radix:
        a = tf.split(a, radix, axis=-1)
        y = tf.keras.layers.Add(name=f"{prefix}splat_out")(
            [o * a for o, a in zip(split, a)]
        )
    else:
        y = tf.keras.layers.Multiply(name=f"{prefix}splat_out")([a, y])
    return y


def resnest_block(
    x,
    n_filter,
    stride_size=1,
    dilation=1,
    group_size=1,
    radix=1,
    block_width=64,
    avd=False,
    avd_first=False,
    downsample=None,
    dropout_rate=0.0,
    expansion=4,
    is_first=False,
    stage=1,
    index=1,
    cv_args: Optional[Dict[str, Union[str, int, float]]] = None,
    bn_args: Optional[Dict[str, Union[str, int, float]]] = None,
):
    if not cv_args:
        cv_args = CV_DEFAULTS
    if not bn_args:
        bn_args = BN_DEFAULTS

    avd = avd and (1 < stride_size or is_first)
    group_width = int(n_filter * (block_width / 64)) * group_size

    y = Conv2D(group_width, 1, name=f"stage{stage}_block{index}_conv1", **cv_args)(x)
    y = BatchNormalization(name=f"stage{stage}_block{index}_bn1", **bn_args)(y)
    if 0 < dropout_rate:
        y = Dropout(dropout_rate, name=f"stage{stage}_block{index}_dropout1")(y)
    y = Activation("relu", name=f"stage{stage}_block{index}_act1")(y)

    if avd:
        avd_layer = AveragePooling2D(
            3,
            strides=stride_size,
            padding="same",
            name=f"stage{stage}_block{index}_avd",
        )
        stride_size = 1
        if avd_first:
            y = avd_layer(y)

    if 0 < radix:
        y = splat_block(
            y,
            group_width,
            3,
            stride_size,
            dilation,
            group_size,
            radix,
            dropout_rate,
            expansion,
            prefix=f"stage{stage}_block{index}",
        )
    else:
        y = Conv2D(
            group_width,
            3,
            strides=stride_size,
            dilation_rate=dilation,
            name=f"stage{stage}_block{index}_conv2",
            **cv_args,
        )(y)
        y = BatchNormalization(name=f"stage{stage}_block{index}_bn2", **bn_args)(y)
        if 0 < dropout_rate:
            y = Dropout(dropout_rate, name=f"stage{stage}_block{index}_dropout2")(y)
        y = Activation("relu", name=f"stage{stage}_block{index}_act2")(y)

    if avd and not avd_first:
        y = avd_layer(y)

    y = Conv2D(
        n_filter * expansion, 1, name=f"stage{stage}_block{index}_conv3", **cv_args
    )(y)
    y = BatchNormalization(name=f"stage{stage}_block{index}_bn3", **bn_args)(y)
    if 0 < dropout_rate:
        y = Dropout(dropout_rate, name=f"stage{stage}_block{index}_dropout3")(y)
    residual = x
    if downsample is not None:
        residual = downsample
    y = Add(name=f"stage{stage}_block{index}_shorcut")([y, residual])
    y = Activation("relu", name=f"stage{stage}_block{index}_shorcut_act")(y)

    return y


def resnest_module(
    x,
    n_filter,
    n_block,
    stride_size=1,
    dilation=1,
    group_size=1,
    radix=1,
    block_width=64,
    avg_down=True,
    avd=False,
    avd_first=False,
    dropout_rate=0.0,
    expansion=4,
    is_first=True,
    stage=1,
    cv_args: Optional[Dict[str, Union[str, int, float]]] = None,
    bn_args: Optional[Dict[str, Union[str, int, float]]] = None,
):
    if not cv_args:
        cv_args = CV_DEFAULTS
    if not bn_args:
        bn_args = BN_DEFAULTS

    downsample = None
    if stride_size != 1 or tf.keras.backend.int_shape(x)[-1] != (n_filter * expansion):
        if avg_down:
            if dilation == 1:
                downsample = AveragePooling2D(
                    stride_size,
                    strides=stride_size,
                    padding="same",
                    name=f"stage{stage}_downsample_avgpool",
                )(x)
            else:
                downsample = AveragePooling2D(
                    1,
                    strides=1,
                    padding="same",
                    name=f"stage{stage}_downsample_avgpool",
                )(x)
            downsample = Conv2D(
                n_filter * expansion,
                1,
                name=f"stage{stage}_downsample_conv1",
                **cv_args,
            )(downsample)
            downsample = BatchNormalization(
                name=f"stage{stage}_downsample_bn1", **bn_args
            )(downsample)
        else:
            downsample = Conv2D(
                n_filter * expansion,
                1,
                strides=stride_size,
                name=f"stage{stage}_downsample_conv1",
                **cv_args,
            )(x)
            downsample = BatchNormalization(
                name=f"stage{stage}_downsample_bn1", **bn_args
            )(downsample)

    if dilation == 1 or dilation == 2 or dilation == 4:
        y = resnest_block(
            x,
            n_filter,
            stride_size,
            2 ** (dilation // 4),
            group_size,
            radix,
            block_width,
            avd,
            avd_first,
            downsample,
            dropout_rate,
            expansion,
            is_first,
            stage=stage,
        )
    else:
        raise ValueError(f"unknown dilation size '{dilation}'")

    for index in range(1, n_block):
        y = resnest_block(
            y,
            n_filter,
            1,
            dilation,
            group_size,
            radix,
            block_width,
            avd,
            avd_first,
            dropout_rate=dropout_rate,
            expansion=expansion,
            stage=stage,
            index=index + 1,
        )
    return y


def ResNet(
    x,
    stack,
    n_class=1000,
    include_top=True,
    dilation=1,
    group_size=1,
    radix=1,
    block_width=64,
    stem_width=64,
    deep_stem=False,
    avg_down=False,
    avd=False,
    avd_first=False,
    dropout_rate=0.0,
    expansion=4,
    pooling: str = "avg",
    classifier_activation: str = "softmax",
    cv_args: Optional[Dict[str, Union[str, int, float]]] = None,
    bn_args: Optional[Dict[str, Union[str, int, float]]] = None,
):
    if not cv_args:
        cv_args = CV_DEFAULTS
    if not bn_args:
        bn_args = BN_DEFAULTS

    if deep_stem:
        y = Conv2D(stem_width, 3, strides=2, name="stem_conv1", **cv_args)(x)
        y = BatchNormalization(name="stem_bn1", **bn_args)(y)
        y = Activation("relu", name="stem_act1")(y)
        y = Conv2D(stem_width, 3, name="stem_conv2", **cv_args)(y)
        y = BatchNormalization(name="stem_bn2", **bn_args)(y)
        y = Activation("relu", name="stem_act2")(y)
        y = Conv2D(stem_width * 2, 3, name="stem_conv3", **cv_args)(y)
        y = BatchNormalization(name="stem_bn3", **bn_args)(y)
        y = Activation("relu", name="stem_act3")(y)
    else:
        y = Conv2D(64, 7, strides=2, name="stem_conv1", **cv_args)(x)
        y = BatchNormalization(name="stem_bn1", **bn_args)(y)
        y = Activation("relu", name="stem_act1")(y)
    y = MaxPool2D(3, strides=2, padding="same", name="stem_pooling")(y)

    # Stage 1
    y = resnest_module(
        y,
        64,
        stack[0],
        1,
        1,
        group_size,
        radix,
        block_width,
        avg_down,
        avd,
        avd_first,
        expansion=expansion,
        is_first=False,
        stage=1,
    )
    # Stage 2
    y = resnest_module(
        y,
        128,
        stack[1],
        2,
        1,
        group_size,
        radix,
        block_width,
        avg_down,
        avd,
        avd_first,
        expansion=expansion,
        stage=2,
    )

    if dilation == 1:
        dilation = [1, 1]
        stride_size = [2, 2]
    elif dilation == 4:
        dilation = [2, 4]
        stride_size = [1, 1]
    elif dilation == 2:
        dilation = [1, 2]
        stride_size = [2, 1]
    else:
        stride_size = [2 if d == 1 else 1 for d in dilation]

    # Stage 3
    y = resnest_module(
        y,
        256,
        stack[2],
        stride_size[0],
        dilation[0],
        group_size,
        radix,
        block_width,
        avg_down,
        avd,
        avd_first,
        dropout_rate,
        expansion,
        stage=3,
    )

    # Stage 4
    y = resnest_module(
        y,
        512,
        stack[3],
        stride_size[1],
        dilation[1],
        group_size,
        radix,
        block_width,
        avg_down,
        avd,
        avd_first,
        dropout_rate,
        expansion,
        stage=4,
    )

    if pooling:
        if pooling == "avg":
            y = GlobalAveragePooling2D(name="avg_pool")(y)
        else:
            y = GlobalMaxPooling2D(name="max_pool")(y)

    if include_top:
        y = Dense(
            n_class,
            activation=classifier_activation,
            name="predictions",
            dtype="float32",
        )(y)

    return y


def ResNeSt(
    model_name: str,
    blocks: List[int],
    default_size: int,
    radix: int = 2,
    group_size: int = 1,
    block_width: int = 64,
    stem_width: int = 64,
    input_tensor: Optional[tf.keras.Input] = None,
    input_shape: Optional[Tuple[int]] = None,
    classes: int = 1000,
    include_top: bool = True,
    weights: str = "imagenet",
    dropout_rate: float = 0,
    dilation: int = 1,  # [1, 2, 4]
    pooling: str = "avg",
    classifier_activation: str = "softmax",
    name: str = None,
):
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
            f"Received: weights={weights}"
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            "If using `weights` as `'imagenet'` with `include_top`"
            " as true, `classes` should be 1000"
            f"Received: classes={classes}"
        )

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=tf.keras.backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if input_tensor is None:
        input_tensor = tf.keras.layers.Input(shape=input_shape)
    if not tf.keras.backend.is_keras_tensor(input_tensor):
        input_tensor = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)

    x = input_tensor
    y = ResNet(
        x,
        blocks,
        classes,
        include_top,
        radix=radix,
        group_size=group_size,
        block_width=block_width,
        stem_width=stem_width,
        deep_stem=True,
        avg_down=True,
        avd=True,
        avd_first=False,
        dropout_rate=dropout_rate,
        dilation=dilation,
        pooling=pooling,
        classifier_activation=classifier_activation,
    )

    inputs = layer_utils.get_source_inputs(input_tensor)

    # Create model.
    model = tf.keras.Model(inputs, y, name=name)

    # Load weights.
    if weights == "imagenet":
        file_url = WEIGHTS_HASHES[model_name]["url"]
        file_hash = WEIGHTS_HASHES[model_name]["hash"]
        file_suffix = ".h5"
        file_name = model_name + file_suffix
        weights_path = data_utils.get_file(
            file_name, file_url, cache_subdir="models", file_hash=file_hash
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def ResNeSt50(
    input_tensor=None,
    input_shape=None,
    classes=1000,
    include_top=True,
    weights: str = "imagenet",
    dropout_rate: float = 0,
    dilation: int = 1,  # [1, 2, 4]
    pooling: str = "avg",
    classifier_activation: str = "softmax",
    name: str = "resnest50",
):
    return ResNeSt(
        "resnest50",
        [3, 4, 6, 3],
        default_size=224,
        stem_width=32,
        input_tensor=input_tensor,
        input_shape=input_shape,
        classes=classes,
        include_top=include_top,
        weights=weights,
        dropout_rate=dropout_rate,
        dilation=dilation,
        pooling=pooling,
        classifier_activation=classifier_activation,
        name=name,
    )


def ResNeSt101(
    input_tensor=None,
    input_shape=None,
    classes=1000,
    include_top=True,
    weights="imagenet",
    dropout_rate: float = 0,
    dilation: int = 1,  # [1, 2, 4]
    pooling: str = "avg",
    classifier_activation: str = "softmax",
    name: str = "resnest101",
):
    return ResNeSt(
        "resnest101",
        [3, 4, 23, 3],
        default_size=256,
        input_tensor=input_tensor,
        input_shape=input_shape,
        classes=classes,
        include_top=include_top,
        weights=weights,
        dropout_rate=dropout_rate,
        dilation=dilation,
        pooling=pooling,
        classifier_activation=classifier_activation,
        name=name,
    )


def ResNeSt200(
    input_tensor=None,
    input_shape=None,
    classes=1000,
    include_top=True,
    weights="imagenet",
    dropout_rate: float = 0,
    dilation: int = 1,  # [1, 2, 4]
    pooling: str = "avg",
    classifier_activation: str = "softmax",
    name: str = "resnest200",
):
    return ResNeSt(
        "resnest200",
        [3, 24, 36, 3],
        default_size=320,
        input_tensor=input_tensor,
        input_shape=input_shape,
        classes=classes,
        include_top=include_top,
        weights=weights,
        dropout_rate=dropout_rate,
        dilation=dilation,
        pooling=pooling,
        classifier_activation=classifier_activation,
        name=name,
    )


def ResNeSt269(
    input_tensor=None,
    input_shape=None,
    classes=1000,
    include_top=True,
    weights="imagenet",
    dropout_rate: float = 0.0,
    dilation: int = 1,  # [1, 2, 4]
    pooling: str = "avg",
    classifier_activation: str = "softmax",
    name: str = "resnest269",
):
    return ResNeSt(
        "resnest269",
        [3, 30, 48, 8],
        default_size=416,
        input_tensor=input_tensor,
        input_shape=input_shape,
        classes=classes,
        include_top=include_top,
        weights=weights,
        dropout_rate=dropout_rate,
        dilation=dilation,
        pooling=pooling,
        classifier_activation=classifier_activation,
        name=name,
    )


__all__ = [
    "ResNeSt50",
    "ResNeSt101",
    "ResNeSt200",
    "ResNeSt269",
]
