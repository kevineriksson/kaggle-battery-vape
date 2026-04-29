import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    # ... load model from args.ckpt ...
    # ... run inference on test set in args.data ...
    # ... build dataframe with the columns Kaggle expects ...

    df.to_csv(args.out, index=False)
    print(f"Wrote submission to {args.out}")

if __name__ == '__main__':
    main()