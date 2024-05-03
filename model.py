import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models import VGG19_Weights
import model_parts as ap
import torch.nn.functional as F

class StyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg19 = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features

        for param in self.vgg19.parameters():
            param.requires_grad = False

        self.vgg19_feature_map = {
            '1': 'relu1_1',
            '6': 'relu2_1',
            '11': 'relu3_1', 
            '20': 'relu4_1'
        }

        self.vgg19_concat_map = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '17': 'relu3_4'
        }

        self.adain = ap.AdaIN()

        self.d4 = ap.VggDecoderBlock(512, 256, 4)
        self.d3 = ap.VggDecoderBlock(256, 128, 3)
        self.d2 = ap.VggDecoderBlock(128, 64, 2)
        self.d1 = ap.VggDecoderBlock(64, 3, 1)

    def encoder(self, input, style_features=None, concat_features=None):
        """
            Because we needs style_features for style representations and concat_features from
            the content image for final image reconstruction, therefore

            style_features: If this is NOT null then the input is style image
            concat_features: If this is NOT null then the input is content image
            IF both of them are null, input is the final reconstructed image and the function will
            return it's features and the encoder output for loss calculation
        """        

        layer = 1
        features = []

        # Cut off the model and sets up hooks is cleaner than looping through modules
        # But I want to use torch.compile(), which doesn't support hooks
        for num, module in self.vgg19.named_modules():
            if num != '' and int(num) <= 20:
                # In the paper the author replaces max pool with avg pool
                if isinstance(module, nn.MaxPool2d):
                    input = F.avg_pool2d(input, kernel_size=2, stride=2)
                else:
                    input = module(input)

                if num in self.vgg19_feature_map:
                    if style_features is not None:
                        style_features.append(input)
                    elif concat_features is None:
                        features.append(input)

                if num in self.vgg19_concat_map and concat_features is not None:
                    concat_features[f"layer{layer}"] = input
                    layer += 1

        if style_features is None and concat_features is None:
            return input, features
        return input

    def forward(self, content, style, training=False):
        concat_features = {}
        style_features = []

        content = self.encoder(content, concat_features=concat_features)
        style = self.encoder(style, style_features=style_features)

        adain = self.adain(content, style)

        x = self.d4(adain, None)
        x = self.d3(x, concat_features["layer3"])
        x = self.d2(x, concat_features["layer2"])
        x = self.d1(x, concat_features["layer1"])
        
        if training:
            vgg_out, vgg_out_features = self.encoder(x)
            return vgg_out, adain, vgg_out_features, style_features
        else:
            return x
