import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        con_mean, con_std = utils.mean_std(x)
        sty_mean, sty_std = utils.mean_std(y)
        return (sty_std * ((x - con_mean) / con_std) + sty_mean)
    
class AdaINLoss(nn.Module):
    def __init__(self, _lambda=7.5):
        super().__init__()
        self._lambda = _lambda

    def contentLoss(self, vgg_out, adain_out):
        return F.mse_loss(vgg_out, adain_out)

    def styleLoss(self, vgg_out_features, style_features):
        mean_sum = 0
        std_sum = 0
        for vgg_out, style in zip(vgg_out_features, style_features):
            vgg_out_mean, vgg_out_std = utils.mean_std(vgg_out)
            style_mean, style_std = utils.mean_std(style)

            mean_sum += F.mse_loss(vgg_out_mean, style_mean)
            std_sum += F.mse_loss(vgg_out_std, style_std)
        return mean_sum + std_sum

    def forward(self, vgg_out, adain_out, vgg_out_features, style_features):
        """
            The input will go through encoder1 -> adain -> decoder -> encoder 2 (for calculating losses)
            vgg_out: Output of encoder2 
            adain_out: Output of the adain layer
            vgg_out_features: Features from encoder 2
            style_features: Features from encoder 1
        """        
        return self.contentLoss(vgg_out, adain_out), self._lambda * self.styleLoss(vgg_out_features, style_features)

class VggDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer):
        super().__init__()

        # In the paper the author uses upsampling instead of conv transpose
        self.up_scale = nn.Upsample(scale_factor=2, mode='nearest')
        # So the default decoder above uses transpose which halve the number of channels
        # But we use upsample here so the number of input channels needs to be doubled except 
        # for layer 4 since we do not concatenate anything at this layer
        if layer == 1 or layer == 2:
            self.seq = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU() 
            )
        elif layer == 3:
            self.seq = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU()
            )
        else: # Layer 4
            self.seq = nn.Sequential(
                # There is no need to concatenate features for layer 4 so upscale is not
                # going to havle the number of features -> in_channels stays the same
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU()
            )

    def forward(self, inp, con_channels):
        if con_channels is not None:
            inp = self.up_scale(inp)
            inp = utils.pad_fetures(inp, con_channels)
            inp = torch.cat([inp, con_channels], dim=1)
        return self.seq(inp)
