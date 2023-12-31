import torch
from torch import nn
import torchvision.models as models

def get_vgg_model(out_features: int):
    """
        ...
    """

    pretrained_vgg_model = models.vgg19_bn(weights='DEFAULT')
    
    print(f"Original output layer: {pretrained_vgg_model.classifier[-1]}")
    IN_FEATURES = pretrained_vgg_model.classifier[-1].in_features
    OUTPUT_DIM = out_features
    final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    pretrained_vgg_model.classifier[-1] = final_fc
    print(f"New output layer: {pretrained_vgg_model.classifier[-1]}")

    return pretrained_vgg_model