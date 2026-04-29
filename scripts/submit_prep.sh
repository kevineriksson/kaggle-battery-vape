#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --account=ealloc_ati-1-deep
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/prep_%j.out

module load python/3.11
source ~/kaggle-env/bin/activate

cd $HOME/kaggle-battery-vape

# Adjust these paths to match what's actually inside data/ after unzipping
python src/prepare_data.py \
    --coco data/annotations.json \
    --images data/images \
    --out data/yolo \
    --val_frac 0.2