import os
import sys

import torch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL
import torchvision.transforms.functional as F

import model

def image_grid(**kwargs):
    col_names = list(kwargs.keys())
    num_rows = len(kwargs[col_names[0]])
    num_cols = len(col_names)
        
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 9))

    if num_rows != 1 and num_cols != 1:
        for row in range(num_rows):
            for col in range(num_cols):
                ax = axs[row, col]
                if row == 0:
                    ax.set_title(col_names[col])
                ax.imshow(kwargs[col_names[col]][row])
                ax.axis('off')
    elif num_rows != 1 and num_cols == 1:
        for row in range(num_rows):
            ax = axs[row]
            if row == 0:
                ax.set_title(col_names[0])
            ax.imshow(kwargs[col_names[0]][row])
            ax.axis('off')
            
    elif num_rows == 1 and num_cols != 1:
        for col in range(num_cols):
            ax = axs[col]
            ax.set_title(col_names[col])
            ax.imshow(kwargs[col_names[col]][0])
            ax.axis('off')
    else:
        axs.set_title(col_names[0])
        axs.imshow(kwargs[col_names[0]][0])
        axs.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def paste_concat_images(input1, input2, output):
    input1 = F.to_pil_image(input1)
    input2 = F.to_pil_image(input2)
    output = F.to_pil_image(output)
    
    wo, ho = output.size
    h2 = int(ho * 0.25)
    aspect_ratio = input2.width / input2.height
    w2 = int(h2 * aspect_ratio)
    
    input2 = input2.resize((w2, h2))
    
    # Bottom left corner
    output.paste(input2, (0, ho - h2))
    
    new = Image.new('RGB', (input1.width + output.width, input1.height))
    new.paste(input1, (0, 0))
    new.paste(output, (input1.width, 0))
    
    return new

def test_style_model(model, con_images_path, sty_images_path, num_samples=8):
    cons = os.listdir(con_images_path)
    cons = np.random.choice(cons, num_samples, replace=False)

    stys = os.listdir(sty_images_path)
    stys = np.random.choice(stys, num_samples, replace=False)

    con_images = []
    sty_images = []
    output = []

    for con, sty in zip(cons, stys):   
        con = os.path.join(con_images_path, con) 
        con = Image.open(con).convert("RGB")
        con_images.append(con)
        con = norm_pil(con)

        sty = os.path.join(sty_images_path, sty) 
        sty = Image.open(sty).convert("RGB")
        sty_images.append(sty)
        sty = norm_pil(sty)

        con = con.unsqueeze(0).float()
        sty = sty.unsqueeze(0).float()

        model.eval()
        with torch.no_grad():
            out = model(con, sty)
        out = denorm(out.squeeze())
        output.append(out.permute(1, 2, 0))

    image_grid(Content=con_images, Style=sty_images, Output=output)            

def save_train_state(model, optimizer, scaler, epoch, path):
    # This one is for resuming training
    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'epoch': epoch
    }, path)

    path = path + ".epoch" + str(epoch + 1)
    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'epoch': epoch
    }, path)

def load_train_state(path):
    try:
        state = torch.load(path)
        return state["model"], state["optimizer"], state["scaler"], state["epoch"]
    except Exception as e:
        print(e)
        sys.exit("Loading train state failed, existing")

def norm(tensor):
    return F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def norm_pil(image):
    ten = F.to_tensor(image)
    return norm(ten)

def denorm(tensor):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def mean_std(input):
    mean = torch.mean(input, dim=(2, 3), keepdims=True)
    """
    For some reasons the torch.std() uses Bessel’s correction by default so unbiased=False 
    is used to make standard deviation "standard". And also keepdims to makes this function
    correctly
    Add 1e-6 to avoid dividing by zero which cause AdaIN's output to contain NaN
    """
    std = torch.std(input, dim=(2, 3), unbiased=False, keepdims=True) + 1e-6
    return mean, std

def pad_fetures(up, con_channels):
    """
    We need to pad the features with 0 when we concatenating upscaled 
    features that were previously downscaled from odd dimension features
    For example: 25 -> down -> 12 -> up -> 24 -> pad -> 25
    """
    diffY = con_channels.size()[2] - up.size()[2]
    diffX = con_channels.size()[3] - up.size()[3]
    up = torch.nn.functional.pad(up, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
    return up

def list_images(folder_path):
    """
    Instead of creating a labels file, we can just pass a list of files to the 
    decoder via files argument. And it does not take too much time either (2s 
    from my testing)
    """
    temp = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            temp.append(filename)
    return temp

def resize_large_images(folder_path):
    # Temporary raise the maximum image pixels to avoid PIL.Image.DecompressionBombError
    PIL.Image.MAX_IMAGE_PIXELS = 933120000

    max_res = 3840 * 2160
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = f"{folder_path}/{filename}"
            image = Image.open(path).convert("RGB")
            tensor = F.pil_to_tensor(image)

            current_res = tensor.shape[1] * tensor.shape[2]

            if current_res > max_res:
                scale = (max_res / current_res) ** 0.5
                tensor = F.resize(tensor, (int(tensor.shape[1] * scale), int(tensor.shape[2] * scale)))
                image = F.to_pil_image(tensor)
                image.save(os.path.join(folder_path, filename))

def remove_corrupted_jpeg(folder_path):
    """
    https://stackoverflow.com/questions/33548956/detect-avoid-premature-end-of-jpeg-in-cv2-python
    This can remove MOST corrupted images from my testing but good enough I guess 
    Also this method is much faster than Image.open(path).convert("RGB") with try-except
    """
    for filename in os.listdir(folder_path):
        path = f"{folder_path}/{filename}"
        with open(path, 'rb') as f:
            check_chars = f.read()[-2:]
            if check_chars != b'\xff\xd9':
                os.remove(path)
