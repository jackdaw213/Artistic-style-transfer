import torch
import os
import torchvision.transforms as transforms
from PIL import Image

import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn

import exinput

"""
So the dataset I'm using for style representation is this one https://www.kaggle.com/c/painter-by-numbers/data
There are some problems with it, the first thing is that there are some gif images with jpg extention which 
DALI does not like. Second, there are some images with monsterous resolution hiding inside the dataset, the 
largest one I found was 30000x29605 - 220MB (⌐■_■) ( •_•)>⌐■-■. These images makes the program crashed with 
OOM error which confused me a bit tbh, thought it was because of a bug in my dali_pipeline
"""
class StyleDataset(torch.utils.data.Dataset):
    def __init__(self, content_dir, style_dir):
        self.content = os.listdir(content_dir)
        self.style = os.listdir(style_dir)
        self.pair = list(zip(self.content, self.style))

        self.content_dir = content_dir
        self.style_dir = style_dir

        self.transform = transforms.Compose([
            transforms.Resize((512), antialias=True),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, index):
        content, style = self.pair[index]

        content = os.path.join(self.content_dir, content)
        style = os.path.join(self.style_dir, style)

        content = Image.open(content).convert("RGB")
        style = Image.open(style).convert("RGB")

        content = self.transform(content)
        style = self.transform(style)

        return content, style
    
    @staticmethod
    @pipeline_def(device_id=0, py_start_method="spawn")
    def dali_pipeline(content_dir, style_dir, bs):
        content_images, style_images = fn.external_source(
            source=exinput.ExternalInputCallable(content_dir, style_dir, bs), 
            num_outputs=2,
            parallel=True, 
            batch=False
        )

        content_images = fn.decoders.image(content_images, device="mixed", output_type=types.RGB)
        style_images = fn.decoders.image(style_images, device="mixed", output_type=types.RGB)

        content_images = fn.resize(content_images, size=512, dtype=types.FLOAT)
        style_images = fn.resize(style_images, size=512, dtype=types.FLOAT)

        """
        Pytorch ToTensor() transform brings the image to [0, 1] range then Normalize() 
        transform normalizes it [1*]. But nvidia dali doesn't have ToTensor() or bring 
        images to [0, 1] range automatically. Therefore we need to multiply the mean and 
        std with 255 [2*]
        [1*]: https://discuss.pytorch.org/t/does-pytorch-automatically-normalizes-image-to-0-1/40022/2
        [2*]: https://github.com/NVIDIA/DALI/issues/4850#issuecomment-1545267530

        Rant: This goddamn bug cost me 5 fricking days to figure it out. FIVE days of 
        looking at other projects online, checking adain/loss function output, checking
        if the model is functioning properly, testing out different model configurations, etc. 
        I was defeated and was about to give up but mama ain't raised a B- losser. So I 
        gathered all of my A+ winner energy and found the above
        """

        content_images = fn.crop_mirror_normalize(content_images, 
                                                dtype=types.FLOAT,
                                                crop=(256, 256),
                                                crop_pos_x=fn.random.uniform(range=(0, 1)),
                                                crop_pos_y=fn.random.uniform(range=(0, 1)),
                                                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        style_images = fn.crop_mirror_normalize(style_images, 
                                                dtype=types.FLOAT,
                                                crop=(256, 256),
                                                crop_pos_x=fn.random.uniform(range=(0, 1)),
                                                crop_pos_y=fn.random.uniform(range=(0, 1)),
                                                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

        return content_images, style_images