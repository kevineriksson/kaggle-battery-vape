#!/bin/bash
#SBATCH --job-name=yolo-pred
#SBATCH --account=ealloc_ati-1-deep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pred_%j.out
#SBATCH --error=logs/pred_%j.err

set -e

module load python/3.11 cuda/12.1
source ~/kaggle-env/bin/activate

cd $HOME/kaggle-battery-vape

mkdir -p submissions
OUT_FILE=submissions/submission_$SLURM_JOB_ID.csv

python src/predict.py \
    --ckpt $CKPT \
    --test_dir $TEST_DIR \
    --out $OUT_FILE \
    --conf 0.25

echo "Wrote: $OUT_FILE"