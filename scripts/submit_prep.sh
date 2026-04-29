#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --account=ealloc_ati-1-deep
#SBATCH --partition=testing
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err

set -e

module load python/3.11
source ~/kaggle-env/bin/activate

cd $HOME/kaggle-battery-vape

python src/prepare_data.py \
    --coco data/train_annotations.json \
    --images data/train/images \
    --out data/yolo \
    --val_frac 0.2