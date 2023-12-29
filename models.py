import torch
from torch import nn
import torchvision.models as models

def get_vgg_model(OUTPUT_DIM: int):
    """
        ...
    """

    pretrained_vgg_model = models.vgg19_bn(weights='DEFAULT')
    IN_FEATURES = pretrained_vgg_model.classifier[-1].in_features
    final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    pretrained_vgg_model.classifier[-1] = final_fc

    return pretrained_vgg_model