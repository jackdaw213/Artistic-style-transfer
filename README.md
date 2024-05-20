# Artistic-style-transfer
Unofficial Pytorch implementation of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization [Huang+, ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)

Original torch implementation from the author can be found [here](https://github.com/xunhuang1995/AdaIN-style).

This implementation uses Nvidia DALI and AMP to accelerate the training process, with WanDB employed for monitoring.

## Prerequisites

1. Clone this repository 

   ```bash
   git clone https://github.com/jackdaw213/Artistic-style-transfer
   cd Artistic-style-transfer
   ```
2. Install Conda and create an environment
    ```shell
    conda create -n artistic_style_transfer
    ```
3. Install all dependencies from requirements.txt
    ```shell
    conda activate artistic_style_transfer
    pip install -r requirements.txt
    ```
This should prepare the Conda environment for both training and testing (pretrained model available below)

## Train

1. Download the [COCO](https://github.com/nightrome/cocostuff) dataset for content images and the [Wikiart](https://www.kaggle.com/c/painter-by-numbers) dataset for style images. Extract the files and organize them into the 'data' folder, with subfolders 'train_content', 'val_content', 'train_style', and 'val_style'.

2. Preprocess the dataset

    WikiArt dataset contains corrupted JPEG images (file ends prematurely) and images with 105x pixel counts of a 4K image. This step should remove MOST of the corrupted images and resize any images with pixel counts higher than 3840 * 2160.

    ```python
    python preprocess.py
    ```

    ```
    preprocess.py [-h]
                  [--train_style TRAIN_STYLE_FOLDER]
                  [--val_style VAL_STYLE_FOLDER]
    ```
3. Train the model.

    ```python
    python train.py --enable_dali --enable_amp --enable_wandb
    ```

    ```
    train.py [-h]
             [--epochs EPOCHS]
             [--batch_size BATCH_SIZE]
             [--num_workers NUM_WORKERS]
             [--train_dir_content TRAIN_DIR_CONTENT]
             [--val_dir_content VAL_DIR_CONTENT]
             [--train_dir_style TRAIN_DIR_STYLE]
             [--val_dir_style VAL_DIR_STYLE]
             [--optimizer OPTIMIZER]
             [--learning_rate LEARNING_RATE]
             [--momentum MOMENTUM]
             [--resume_id RESUME_ID]
             [--checkpoint_freq CHECKPOINT_FREQ]
             [--amp_dtype AMP_DTYPE]
             [--enable_dali]
             [--enable_amp]
             [--enable_wandb]
    ```

    The model was trained on an RTX 3080 10G for 10 epoches with Dataloader and AMP.

    | Training setup      | Batch size | GPU memory usage | Training time |
    |---------------------|------------|------------------|---------------|
    | DALI                | 4          | 6GB              | 3.8 hours     |
    | DALI + AMP          | 8          | 6.5GB            | 2.2 hours     |
    | DataLoader          | 8          | 9GB              | 4.4 hours     |
    | DataLoader + AMP    | 8          | 4GB              | 2.4 hours     |

    WARNING: Nvidia DALI only supports Nvidia GPUs. BFloat16 is supported only on RTX 3000/Ampere GPUs and above, while GPU Direct Storage (GDS) is supported only on server-class GPUs. Using Float16 might cause NaN loss during training, whereas BFloat16 does not.

    

## Test

1. Download the pretrained model [here](https://drive.google.com/file/d/1m3izs7WCyKVY0hbAER7q4F6OjNPcJZyV/view?usp=sharing) and put it in the model folder

2. Generate the output image using the command bellow.

    ```python
    python test -c content_image_path -s style_image_path
    ```

    ```
    test.py [-h] 
            [--content CONTENT] 
            [--style STYLE]
            [--model MODEL_PATH] 
    ```

## Result

![image](https://github.com/jackdaw213/Artistic-style-transfer/blob/master/img/comp.jpg)
![image](https://github.com/jackdaw213/Artistic-style-transfer/blob/master/img/comp1.jpg)
![image](https://github.com/jackdaw213/Artistic-style-transfer/blob/master/img/comp2.jpg)
![image](https://github.com/jackdaw213/Artistic-style-transfer/blob/master/img/comp3.jpg)
![image](https://github.com/jackdaw213/Artistic-style-transfer/blob/master/img/comp5.jpg)

## References

- [X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)
- [Original implementation in Torch.](https://github.com/xunhuang1995/AdaIN-style) 
- [Irasin's implementation.](https://github.com/irasin/Pytorch_AdaIN)

