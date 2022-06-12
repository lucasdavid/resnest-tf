from collections import OrderedDict
from typing import List, Optional, Tuple

import tensorflow as tf

from .resnet import ResNet


def _build_input_tensor(input_tensor, input_shape):
    if input_tensor is None:
        return tf.keras.layers.Input(shape=input_shape)
    if not tf.keras.backend.is_keras_tensor(input_tensor):
        return tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
    return input_tensor


def ResNeSt(
    weights_id: str,
    blocks: List[int],
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
    name: str = "resnest50",
):
    x = _build_input_tensor(input_tensor, input_shape)
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
    model = tf.keras.Model(x, y, name=name)

    # if weights == "imagenet":
    #     _load_weights(model, WEIGHTS[weights_id])
    # elif weights is not None:
    #    model.load_weights(weights)
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
        "resnest101",
        [3, 24, 36, 3],
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
