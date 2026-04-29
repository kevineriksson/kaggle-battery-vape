#!/bin/bash
#SBATCH --job-name=yolo-train
#SBATCH --account=ealloc_ati-1-deep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e

module load python/3.11 cuda/12.1
source ~/kaggle-env/bin/activate

cd $HOME/kaggle-battery-vape
echo "Commit: $(git rev-parse HEAD)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi -L)"

OUT=runs/$SLURM_JOB_ID
mkdir -p $OUT

python src/train.py \
    --data data/yolo/data.yaml \
    --out $OUT \
    --model yolov8s.pt \
    --epochs 150 \
    --batch 16 \
    --imgsz 640