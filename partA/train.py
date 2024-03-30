import os
import sys
import torch
import random
import logging
import argparse
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def main(args: argparse.Namespace):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-uw",
                        "--use_wandb",
                        type=str,
                        default="false",
                        help="Use Weights and Biases or not; [true, false]")
    parser.add_argument("-dp",
                        "--dataset_path",
                        type=str,
                        default="dataset/inaturalist_12K/",
                        help="Path to downloaded data")
    args = parser.parse_args()
    logging.info(args)

    # prepare data
    dataset_path = os.path.abspath(args.dataset_path)
    sets = ["train", "test"]
    classes = [f for f in os.listdir(os.path.join(dataset_path, sets[0])) if not f.startswith(".")]
    labels_to_idx = {c:idx for idx, c in enumerate(classes)}
    idx_to_labels = {v:k for k,v in labels_to_idx.items()}
    
    train_percent = 0.8
    xtrain, ytrain = [], []
    xvalid, yvalid = [], []
    xtest, ytest = [], []
    for set in sets:
        for cls in classes:
            img_dir = os.path.join(dataset_path, set, cls)
            images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if not f.startswith(".")]
            n_images = len(images)
            if set == "train":
                k = int(n_images * train_percent)
                random.shuffle(images)
                xtrain += images[:k]
                ytrain += [labels_to_idx[cls]] * k
                xvalid += images[k:n_images]
                yvalid += [labels_to_idx[cls]] * (n_images - k)
            elif set == "test":
                xtest += images
                ytest += [labels_to_idx[cls]] * n_images

    train_dataset = ImageDataset(xtrain, ytrain, Resize((256,256)))
    valid_dataset = ImageDataset(xvalid, yvalid, Resize((256,256)))
    test_dataset = ImageDataset(xtest, ytest, Resize((256,256)))

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    main(args)
