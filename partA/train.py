import os
import sys
import torch
import random
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader

from model import ConvNeuralNet

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
        image = image / 255
        if image.shape[0] == 1:       # handle gray scale images
            image = image.repeat(3,1,1)
        if self.transform:
            image = self.transform(image)
        return image, label

def main(args: argparse.Namespace):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConvNeuralNet(
        in_dims=args.in_dims,
        out_dims=10,
        conv_activation="relu",
        dense_activation="relu",
        n_filters=256,
        filter_org="half",
        data_aug=True,  # argparse will take, won't be required
        batch_norm=True,
        dropout=0.1,
    )

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.005)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1)

    n_epochs = 20
    for epoch in range(n_epochs):
        acc = 0
        cnt = 0
        for xb, yb in train_dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_pred = model(xb)
            acc += (torch.argmax(y_pred, 1) == yb).float().sum()
            cnt += len(yb)
            loss = loss_fn(y_pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        acc /= cnt

        val_acc = 0
        cnt = 0
        with torch.no_grad():
            for xb, yb in valid_dataloader:
                xb = xb.to(device)
                yb = yb.to(device)
                y_pred = model(xb)
                val_acc += (torch.argmax(y_pred, 1) == yb).float().sum()
                cnt += len(yb)
        val_acc /= cnt

        print("Epoch %d: valid accuracy %.2f%% train accuracy %.2f%% " % (epoch, val_acc*100, acc*100))
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-uw",
                        "--use_wandb",
                        default=False,
                        action="store_true",
                        help="Use Weights and Biases or not")
    parser.add_argument("-dp",
                        "--dataset_path",
                        type=str,
                        default="dataset/inaturalist_12K/",
                        help="Path to downloaded data")
    parser.add_argument("-in",
                        "--in_dims",
                        type=int,
                        default=256,
                        help="Input dimensions of images")
    parser.add_argument("-bn",
                        "--use_batch_norm",
                        default=False,
                        action="store_true",
                        help="Use Batch Normalization or not")
    parser.add_argument("-bs",
                        "--batch_size",
                        type=int,
                        default=64,
                        help="Batch Size for training")
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

    transform = transforms.Compose([
        transforms.Resize((args.in_dims, args.in_dims)),
        transforms.RandomResizedCrop(size=(args.in_dims, args.in_dims)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageDataset(xtrain, ytrain, transform=transform)
    valid_dataset = ImageDataset(xvalid, yvalid, Resize((args.in_dims, args.in_dims)))
    test_dataset = ImageDataset(xtest, ytest, Resize((args.in_dims, args.in_dims)))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    main(args)
