#!/bin/bash
#SBATCH --job-name=wikiart-gan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# EDIT for your cluster: add e.g. #SBATCH --partition=gpu  /  #SBATCH --account=...

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# EDIT: activate your environment
# module load anaconda3 cuda
# conda activate gan

WIKIART_DIR="${WIKIART_DIR:?set WIKIART_DIR=/path/to/public/dir/wikiart}"
SIZE="${SIZE:-64}"
RUN="${RUN:-runs/dcgan${SIZE}}"

# 1) preprocess once (skipped if the .npy already exists)
if [ ! -f "data/wikiart_${SIZE}.npy" ]; then
  python prepare_data.py --src "$WIKIART_DIR" --size "$SIZE" --workers "${SLURM_CPUS_PER_TASK:-8}"
fi

# 2) train
python train.py --data "data/wikiart_${SIZE}.npy" --out "$RUN" --epochs "${EPOCHS:-30}" ${TRAIN_ARGS:-}

# 3) evaluate (FID + IS)
python evaluate.py --ckpt "$RUN/last.pt" --n_samples 10000 --real_is --n_individual 16
