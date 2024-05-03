import torch
import torchvision.transforms.functional as F
import torch.nn.functional as f
import matplotlib.pyplot as plt
import argparse

from PIL import Image

import model
import utils

parser = argparse.ArgumentParser(description='Photorealistic style transfer')

parser.add_argument('-c', '--content', type=str,
                    help='Path to the content image')
parser.add_argument('-s', '--style', type=str,
                    help='Path to the style image')
parser.add_argument('-m', '--model', type=str,
                    help='Path to the model')

args = parser.parse_args()

model = torch.load(args.model)
model.eval()

content = F.to_tensor(Image.open(args.content).convert("RGB"))
content = utils.norm_pil(content)

style = F.to_tensor(Image.open(args.style).convert("RGB"))
style = utils.norm_pil(style)

with torch.no_grad():
    output = model(content, style)

output = utils.denorm(output.squeeze())

plt.imshow(output.permute(1, 2, 0))
plt.show()
