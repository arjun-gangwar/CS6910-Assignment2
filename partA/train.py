import torch
import argparse
import numpy as np

def main(args: argparse.Namespace):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-uw",
                        "--use_wandb",
                        type=str,
                        default="false",
                        help="Use Weights and Biases or not; [true, false]")
    args = parser.parse_args()
    main(args)
