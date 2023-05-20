from __future__ import annotations

from typing import Any, NamedTuple

from torchvision.models import (  # type: ignore[import]
    EfficientNet_B1_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    RegNet_X_1_6GF_Weights,
    RegNet_X_3_2GF_Weights,
    RegNet_X_8GF_Weights,
    RegNet_X_16GF_Weights,
    RegNet_X_32GF_Weights,
    RegNet_X_400MF_Weights,
    RegNet_X_800MF_Weights,
    RegNet_Y_3_2GF_Weights,
    RegNet_Y_16GF_Weights,
    RegNet_Y_32GF_Weights,
    RegNet_Y_400MF_Weights,
    RegNet_Y_800MF_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
    ViT_B_16_Weights,
    ViT_H_14_Weights,
    ViT_L_16_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
    efficientnet_b1,
    mobilenet_v2,
    mobilenet_v3_large,
    regnet_x_1_6gf,
    regnet_x_3_2gf,
    regnet_x_8gf,
    regnet_x_16gf,
    regnet_x_32gf,
    regnet_x_400mf,
    regnet_x_800mf,
    regnet_y_3_2gf,
    regnet_y_16gf,
    regnet_y_32gf,
    regnet_y_400mf,
    regnet_y_800mf,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    vit_b_16,
    vit_h_14,
    vit_l_16,
    wide_resnet50_2,
    wide_resnet101_2,
)


class ModelWeights(NamedTuple):
    constructor: Any
    weights_a: Any
    weights_b: Any


MODELS_AND_WEIGHTS = {
    "efficientnet_b1": ModelWeights(
        efficientnet_b1,
        EfficientNet_B1_Weights.IMAGENET1K_V2,
        EfficientNet_B1_Weights.IMAGENET1K_V1,
    ),
    "mobilenet_v2": ModelWeights(
        mobilenet_v2,
        MobileNet_V2_Weights.IMAGENET1K_V2,
        MobileNet_V2_Weights.IMAGENET1K_V1,
    ),
    "mobilenet_v3_large": ModelWeights(
        mobilenet_v3_large,
        MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    ),
    "regnet_x_8gf": ModelWeights(
        regnet_x_8gf,
        RegNet_X_8GF_Weights.IMAGENET1K_V2,
        RegNet_X_8GF_Weights.IMAGENET1K_V1,
    ),
    "regnet_x_16gf": ModelWeights(
        regnet_x_16gf,
        RegNet_X_16GF_Weights.IMAGENET1K_V2,
        RegNet_X_16GF_Weights.IMAGENET1K_V1,
    ),
    "regnet_x_32gf": ModelWeights(
        regnet_x_32gf,
        RegNet_X_32GF_Weights.IMAGENET1K_V2,
        RegNet_X_32GF_Weights.IMAGENET1K_V1,
    ),
    "regnet_x_400mf": ModelWeights(
        regnet_x_400mf,
        RegNet_X_400MF_Weights.IMAGENET1K_V2,
        RegNet_X_400MF_Weights.IMAGENET1K_V1,
    ),
    "regnet_x_800mf": ModelWeights(
        regnet_x_800mf,
        RegNet_X_800MF_Weights.IMAGENET1K_V2,
        RegNet_X_800MF_Weights.IMAGENET1K_V1,
    ),
    "regnet_x_1_6gf": ModelWeights(
        regnet_x_1_6gf,
        RegNet_X_1_6GF_Weights.IMAGENET1K_V2,
        RegNet_X_1_6GF_Weights.IMAGENET1K_V1,
    ),
    "regnet_x_3_2gf": ModelWeights(
        regnet_x_3_2gf,
        RegNet_X_3_2GF_Weights.IMAGENET1K_V2,
        RegNet_X_3_2GF_Weights.IMAGENET1K_V1,
    ),
    "regnet_y_16gf": ModelWeights(
        regnet_y_16gf,
        RegNet_Y_16GF_Weights.IMAGENET1K_V2,
        RegNet_Y_16GF_Weights.IMAGENET1K_V1,
    ),
    "regnet_y_32gf": ModelWeights(
        regnet_y_32gf,
        RegNet_Y_32GF_Weights.IMAGENET1K_V2,
        RegNet_Y_32GF_Weights.IMAGENET1K_V1,
    ),
    "regnet_y_3_2gf": ModelWeights(
        regnet_y_3_2gf,
        RegNet_Y_3_2GF_Weights.IMAGENET1K_V2,
        RegNet_Y_3_2GF_Weights.IMAGENET1K_V1,
    ),
    "regnet_y_400mf": ModelWeights(
        regnet_y_400mf,
        RegNet_Y_400MF_Weights.IMAGENET1K_V2,
        RegNet_Y_400MF_Weights.IMAGENET1K_V1,
    ),
    "regnet_y_800mf": ModelWeights(
        regnet_y_800mf,
        RegNet_Y_800MF_Weights.IMAGENET1K_V2,
        RegNet_Y_800MF_Weights.IMAGENET1K_V1,
    ),
    "resnext101_32x8d": ModelWeights(
        resnext101_32x8d,
        ResNeXt101_32X8D_Weights.IMAGENET1K_V2,
        ResNeXt101_32X8D_Weights.IMAGENET1K_V1,
    ),
    "resnext50_32x4d": ModelWeights(
        resnext50_32x4d,
        ResNeXt50_32X4D_Weights.IMAGENET1K_V2,
        ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
    ),
    "resnet50": ModelWeights(
        resnet50,
        ResNet50_Weights.IMAGENET1K_V2,
        ResNet50_Weights.IMAGENET1K_V1,
    ),
    "resnet101": ModelWeights(
        resnet101,
        ResNet101_Weights.IMAGENET1K_V2,
        ResNet101_Weights.IMAGENET1K_V1,
    ),
    "resnet152": ModelWeights(
        resnet152,
        ResNet152_Weights.IMAGENET1K_V2,
        ResNet152_Weights.IMAGENET1K_V1,
    ),
    "wide_resnet50_2": ModelWeights(
        wide_resnet50_2,
        Wide_ResNet50_2_Weights.IMAGENET1K_V2,
        Wide_ResNet50_2_Weights.IMAGENET1K_V1,
    ),
    "wide_resnet101_2": ModelWeights(
        wide_resnet101_2,
        Wide_ResNet101_2_Weights.IMAGENET1K_V2,
        Wide_ResNet101_2_Weights.IMAGENET1K_V1,
    ),
    "vit_b_16": ModelWeights(
        vit_b_16,
        ViT_B_16_Weights.IMAGENET1K_V1,
        ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1,
    ),
    "vit_l_16": ModelWeights(
        vit_l_16,
        ViT_L_16_Weights.IMAGENET1K_V1,
        ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1,
    ),
}
