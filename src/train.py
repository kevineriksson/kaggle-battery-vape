import argparse
import os
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # ... your training code ...
    # save checkpoints to args.out every epoch

if __name__ == '__main__':
    main()