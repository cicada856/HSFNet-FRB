from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers


def conv_bn_relu(x, filters, kernel_size, strides=1, name=None):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False, name=None if name is None else name + "_conv")(x)
    x = layers.BatchNormalization(name=None if name is None else name + "_bn")(x)
    return layers.Activation("relu", name=None if name is None else name + "_relu")(x)


def residual_block(x, filters, strides=1, name=None):
    shortcut = x
    x = conv_bn_relu(x, filters, 3, strides=strides, name=None if name is None else name + "_1")
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=None if name is None else name + "_2_conv")(x)
    x = layers.BatchNormalization(name=None if name is None else name + "_2_bn")(x)

    if shortcut.shape[-1] != filters or strides != 1:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False, name=None if name is None else name + "_proj_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=None if name is None else name + "_proj_bn")(shortcut)

    x = layers.Add(name=None if name is None else name + "_add")([x, shortcut])
    return layers.Activation("relu", name=None if name is None else name + "_out")(x)


def build_simple_cnn(input_shape=(256, 256, 1), num_classes=5, dropout=0.25):
    inputs = layers.Input(shape=input_shape)
    x = conv_bn_relu(inputs, 32, 3)
    x = layers.MaxPooling2D(2)(x)
    x = conv_bn_relu(x, 32, 3)
    x = layers.MaxPooling2D(2)(x)
    x = conv_bn_relu(x, 64, 5)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="simple_cnn")


def build_resnet(input_shape=(256, 256, 1), num_classes=5, blocks=(1, 1, 1, 1), filters=(32, 64, 128, 256), dropout=0.25, name="resnet"):
    inputs = layers.Input(shape=input_shape)
    x = conv_bn_relu(inputs, filters[0], 7, strides=2, name="stem")
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="stem_pool")(x)

    for stage_idx, (n_blocks, n_filters) in enumerate(zip(blocks, filters)):
        for block_idx in range(n_blocks):
            strides = 2 if stage_idx > 0 and block_idx == 0 else 1
            x = residual_block(x, n_filters, strides=strides, name="s%d_b%d" % (stage_idx + 1, block_idx + 1))

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name=name)


def build_efficientnetb0(input_shape=(256, 256, 1), num_classes=5, dropout=0.25):
    inputs = layers.Input(shape=input_shape)
    rgb = layers.Concatenate(name="gray_to_rgb")([inputs, inputs, inputs])
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_tensor=rgb)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(base.output)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="efficientnetb0_gray")


def build_keras_resnet50(input_shape=(256, 256, 1), num_classes=5, dropout=0.25):
    inputs = layers.Input(shape=input_shape)
    rgb = layers.Concatenate(name="gray_to_rgb")([inputs, inputs, inputs])
    base = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=rgb)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(base.output)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="resnet50_gray")


def convnext_block(x, dim, mlp_ratio=4, dropout=0.0, name=None):
    shortcut = x
    x = layers.DepthwiseConv2D(7, padding="same", name=None if name is None else name + "_dwconv")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=None if name is None else name + "_ln")(x)
    x = layers.Dense(dim * mlp_ratio, activation=tf.nn.gelu, name=None if name is None else name + "_pw1")(x)
    x = layers.Dropout(dropout, name=None if name is None else name + "_drop1")(x)
    x = layers.Dense(dim, name=None if name is None else name + "_pw2")(x)
    x = layers.Dropout(dropout, name=None if name is None else name + "_drop2")(x)
    return layers.Add(name=None if name is None else name + "_add")([shortcut, x])


def convnext_downsample(x, dim, stride, name):
    x = layers.LayerNormalization(epsilon=1e-6, name=name + "_ln")(x)
    x = layers.Conv2D(dim, 2 if stride == 2 else 4, strides=stride, padding="same", name=name + "_conv")(x)
    return x


def build_convnext_tiny(input_shape=(256, 256, 1), num_classes=5, dropout=0.15):
    """Small ConvNeXt-style CNN backbone for 256x256 FRB dynamic spectra."""
    inputs = layers.Input(shape=input_shape)
    dims = [64, 128, 256, 512]
    depths = [2, 2, 4, 2]

    x = layers.Conv2D(dims[0], 4, strides=4, padding="same", name="stem_conv")(inputs)
    x = layers.LayerNormalization(epsilon=1e-6, name="stem_ln")(x)

    for stage_idx, (dim, depth) in enumerate(zip(dims, depths)):
        if stage_idx > 0:
            x = convnext_downsample(x, dim, stride=2, name="downsample_%d" % stage_idx)
        for block_idx in range(depth):
            x = convnext_block(
                x,
                dim=dim,
                mlp_ratio=4,
                dropout=dropout,
                name="stage%d_block%d" % (stage_idx + 1, block_idx + 1),
            )

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="head_norm")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="convnext_tiny")


def available_backbones():
    return ["simple_cnn", "resnet_small", "resnet18", "resnet50", "efficientnetb0", "convnext_tiny"]


def build_model(backbone, input_shape=(256, 256, 1), num_classes=5, dropout=0.25):
    backbone = backbone.lower()
    if backbone == "simple_cnn":
        return build_simple_cnn(input_shape=input_shape, num_classes=num_classes, dropout=dropout)
    if backbone == "resnet_small":
        return build_resnet(input_shape=input_shape, num_classes=num_classes, blocks=(1, 1, 1, 1), dropout=dropout, name="resnet_small")
    if backbone == "resnet18":
        return build_resnet(input_shape=input_shape, num_classes=num_classes, blocks=(2, 2, 2, 2), dropout=dropout, name="resnet18")
    if backbone == "resnet50":
        return build_keras_resnet50(input_shape=input_shape, num_classes=num_classes, dropout=dropout)
    if backbone == "efficientnetb0":
        return build_efficientnetb0(input_shape=input_shape, num_classes=num_classes, dropout=dropout)
    if backbone == "convnext_tiny":
        return build_convnext_tiny(input_shape=input_shape, num_classes=num_classes, dropout=dropout)
    raise ValueError("Unknown backbone %s. Available: %s" % (backbone, ", ".join(available_backbones())))
