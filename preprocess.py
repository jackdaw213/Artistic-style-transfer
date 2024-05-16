import utils
import argparse

parser = argparse.ArgumentParser(description='Preprocessing WikiArt data')

parser.add_argument('-ts', '--train_style', type=str,
                    default="data/train_style",
                    help='Path to WikiArt train folder')
parser.add_argument('-vs', '--val_style', type=str,
                    default="data/val_style",
                    help='Path to WikiArt val folder')

args = parser.parse_args()

print("Removing corrupt images")
utils.remove_corrupted_jpeg(args.train_style)
utils.remove_corrupted_jpeg(args.val_style)

print("Resizing large images")
utils.resize_large_images(args.train_style)
utils.resize_large_images(args.val_style)
