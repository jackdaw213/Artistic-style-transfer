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
                    default="model/artistic.pt",
                    help='Path to the model')

args = parser.parse_args()

model = torch.jit.load(args.model)
model.eval()

content = Image.open(args.content).convert("RGB")
content = utils.norm_pil(content).unsqueeze(dim=0).float()

style = Image.open(args.style).convert("RGB")
style = utils.norm_pil(style).unsqueeze(dim=0).float()

with torch.no_grad():
    output = model(content, style)

output = utils.denorm(output.squeeze())
F.to_pil_image(output).save("output.jpg")

image = utils.paste_concat_images(utils.denorm(content.squeeze()), 
                                  utils.denorm(style.squeeze()), 
                                  output.squeeze())
image.save("comp.jpg")

plt.imshow(image)
plt.show()
