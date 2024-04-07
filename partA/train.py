import os
import sys
import wandb
import torch
import random
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
from helper import list_of_ints, ImageDataset
from model import ConvNeuralNet

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def train(model, loss_fn, optimizer, n_epochs, train_dataloader, valid_dataloader, use_wandb):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for epoch in range(n_epochs):
        train_acc = 0
        train_loss = []
        cnt = 0
        for xb, yb in train_dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_pred = model(xb)
            train_acc += (torch.argmax(y_pred, 1) == yb).float().sum()
            cnt += len(yb)
            loss = loss_fn(y_pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # log loss
            train_loss.append(loss.detach().item())
        train_acc /= cnt
        train_loss = np.array(train_loss).mean()

        valid_acc = 0
        valid_loss = []
        cnt = 0
        with torch.no_grad():
            for xb, yb in valid_dataloader:
                xb = xb.to(device)
                yb = yb.to(device)
                y_pred = model(xb)
                valid_acc += (torch.argmax(y_pred, 1) == yb).float().sum()
                cnt += len(yb)
                # log loss
                valid_loss.append(loss.detach().item())
            valid_acc /= cnt
            valid_loss = np.array(valid_loss).mean()

        print("Epoch %d: valid accuracy %.2f%% train accuracy %.2f%% train loss %.4f valid loss %.4f" % (epoch+1, valid_acc*100, train_acc*100, train_loss, valid_loss))

        if use_wandb:
            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_acc': train_acc*100,
                'valid_acc': valid_acc*100
            })
    del loss
    torch.cuda.empty_cache()

def wandb_sweep():
    with wandb.init() as run:
        config = wandb.config
        in_dims = config.in_dims
        n_epochs = config.n_epochs
        learning_rate = config.learning_rate
        weight_decay = config.weight_decay
        batch_size = config.batch_size
        conv_activation = config.conv_activation
        dense_activation = config.dense_activation
        dense_size = config.dense_size
        filter_size = config.filter_size
        n_filters = config.n_filters
        filter_org = config.filter_org
        batch_norm = config.batch_norm
        dropout = config.dropout
        data_aug = config.data_aug

        run_name=f"bs_{batch_size}_ca_{conv_activation}_fs_{filter_size}_nf_{n_filters}_fo_{filter_org}_ep_{n_epochs}"
        wandb.run.name=run_name

        if data_aug:
            transform = transforms.Compose([
                transforms.Resize((in_dims, in_dims)),
                transforms.RandomResizedCrop(size=(in_dims, in_dims)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Resize((in_dims, in_dims))

        train_dataset = ImageDataset(xtrain, ytrain, transform=transform)
        valid_dataset = ImageDataset(xvalid, yvalid, Resize((in_dims, in_dims)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        model = ConvNeuralNet(
            in_dims=in_dims,
            out_dims=len(classes),
            conv_activation=conv_activation,
            dense_activation=dense_activation,
            dense_size=dense_size,
            filter_size=filter_size,
            n_filters=n_filters,
            filter_org=filter_org,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=weight_decay)
        train(model, loss_fn, optimizer, n_epochs, train_dataloader, valid_dataloader, True)

def main(args: argparse.Namespace):

    if args.use_wandb:
        wandb.login()
        sweep_config = {
            'method': 'bayes',
            'name' : 'cnn sweeps',
            'metric': {
                'name': 'valid_acc',
                'goal': 'maximize'
            },
            'parameters': {
                'in_dims': {
                    'values': [256]
                },'n_epochs': {
                    'values': [10,15]
                },'learning_rate': {
                    'values': [1e-3, 1e-4]
                },'weight_decay': {
                    'values': [0,0.005,0.5]
                },'batch_size':{
                    'values': [64,128]
                },'conv_activation':{
                    'values': ['relu','gelu','silu','mish']
                },'dense_activation':{
                    'values': ['relu']
                },'dense_size':{
                    'values': [256, 512, 1024]
                },'filter_size':{
                    'values': [[7,5,5,3,3], [3,3,5,5,7], [3,3,3,3,3], [5,5,5,5,5], [7,7,7,7,7]]
                },'n_filters':{
                    'values': [32, 64, 128]
                },'filter_org':{
                    'values': ['same', 'double', 'halve']
                },'batch_norm':{
                    'values': [True, False]
                },'dropout':{
                    'values': [0, 0.2, 0.3]
                },'data_aug':{
                    'values': [True, False]
                }
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, function=wandb_sweep, count=250)
        wandb.finish()
    else:       # not using wandb
        if args.data_aug:
            transform = transforms.Compose([
                transforms.Resize((args.in_dims, args.in_dims)),
                transforms.RandomResizedCrop(size=(args.in_dims, args.in_dims)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Resize((args.in_dims, args.in_dims))

        train_dataset = ImageDataset(xtrain, ytrain, transform=transform)
        valid_dataset = ImageDataset(xvalid, yvalid, Resize((args.in_dims, args.in_dims)))
        test_dataset = ImageDataset(xtest, ytest, Resize((args.in_dims, args.in_dims)))

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        model = ConvNeuralNet(
            in_dims=args.in_dims,
            out_dims=len(classes),
            conv_activation=args.conv_activation,
            dense_activation=args.dense_activation,
            dense_size=args.dense_size,
            filter_size=args.filter_size,
            n_filters=args.n_filters,
            filter_org=args.filter_org,
            batch_norm=args.batch_norm,
            dropout=args.dropout,
        )
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.005)
        n_epochs = 20

        train(model, loss_fn, optimizer, n_epochs, train_dataloader, valid_dataloader, False)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-uw",
                        "--use_wandb",
                        default=False,
                        action="store_true",
                        help="Use Weights and Biases or not")
    parser.add_argument("-wp", 
                        "--wandb_project", 
                        type=str, 
                        default="CS6910-Assignment2",
                        help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we",
                        "--wandb_entity", 
                        type=str,
                        default="arjungangwar",
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
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
                        "--batch_norm",
                        default=False,
                        action="store_true",
                        help="Use Batch Normalization or not")
    parser.add_argument("-bs",
                        "--batch_size",
                        type=int,
                        default=64,
                        help="Batch Size for training")
    parser.add_argument("-dg",
                        "--data_aug",
                        default=False,
                        action="store_true",
                        help="Use Data Augmentation or not")
    parser.add_argument("-ca",
                        "--conv_activation",
                        type=str,
                        default="relu",
                        help="Convolution activation")
    parser.add_argument("-da",
                        "--dense_activation",
                        type=str,
                        default="relu",
                        help="Dense activation")
    parser.add_argument("-ds",
                        "--dense_size",
                        type=int,
                        default=512,
                        help="Dense layer size")
    parser.add_argument("-fs",
                        "--filter_size",
                        type=list_of_ints,
                        default=[7,5,5,3,3],
                        help="Layerwise Filter Size")
    parser.add_argument("-nf",
                        "--n_filters",
                        type=int,
                        default=16,
                        help="Number of filter in first layer")
    parser.add_argument("-fo",
                        "--filter_org",
                        type=str,
                        default="double",
                        help="Filter Organization")
    parser.add_argument("-do",
                        "--dropout",
                        type=int,
                        default=0,
                        help="Dropout for dense layer")
    
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

    main(args)
