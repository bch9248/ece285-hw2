# HW2 — DCGAN on WikiArt (FID / IS with PyTorch-Ignite)

Train a GAN on the [WikiArt dataset](https://www.kaggle.com/datasets/steubk/wikiart), generate paintings,
and evaluate with FID and Inception Score using the
[Ignite approach](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).

```
prepare_data.py              raw WikiArt JPEGs -> data/wikiart_<size>.npy (resized once, fast training)
models.py                    DCGAN generator / discriminator (64x64 or 128x128, optional spectral norm)
train.py                     training loop; logs losses.csv, sample grids per epoch, checkpoints
plot_losses.py               loss curves (png + pdf for LaTeX)
evaluate.py                  FID + IS (Ignite), generated / real image grids
scripts/download_wikiart.sh  Kaggle download + unzip into a shared directory
scripts/fetch_inception_weights.py  pre-cache InceptionV3 weights (for offline compute nodes)
scripts/run_all.sbatch       Slurm job: preprocess -> train -> evaluate
```

## 1. Environment
```bash
conda create -n gan python=3.11 -y && conda activate gan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your cluster's CUDA
pip install -r requirements.txt
python scripts/fetch_inception_weights.py      # on a node with internet
```

## 2. Dataset
Put a Kaggle API token at `~/.kaggle/kaggle.json` (Kaggle -> Settings -> API -> Create New Token), `chmod 600` it, then:
```bash
bash scripts/download_wikiart.sh /path/to/public/datasets
# -> /path/to/public/datasets/wikiart/<Style_Name>/*.jpg
```

## 3. Preprocess (once)
```bash
python prepare_data.py --src /path/to/public/datasets/wikiart --size 64
# optional: a single style often gives cleaner samples
python prepare_data.py --src .../wikiart --size 64 --styles Impressionism --out data/impressionism_64.npy
```

## 4. Train
```bash
python train.py --data data/wikiart_64.npy --out runs/dcgan64 --epochs 30
# variants: --sn (spectral norm in D), --loss hinge, --batch_size 64, --lr_d 1e-4
# resume / extend: python train.py --resume runs/dcgan64/last.pt --epochs 50
```
Or on Slurm (edit partition / env lines first):
```bash
WIKIART_DIR=/path/to/public/datasets/wikiart sbatch scripts/run_all.sbatch
```

## 5. Evaluate
```bash
python evaluate.py --ckpt runs/dcgan64/last.pt --n_samples 10000 --real_is --n_individual 16
python plot_losses.py --run runs/dcgan64      # also done automatically at the end of training
```

## What goes into the report (2 pages, NeurIPS format)
| Requirement | File |
|---|---|
| 1) Hyperparameters | `runs/<run>/args.json` |
| 2) Training loss curves | `runs/<run>/loss_curves.pdf` |
| 3) IS and FID | `runs/<run>/eval_<ckpt>/metrics.json` |
| 4) Generated examples | `runs/<run>/eval_<ckpt>/generated_grid.png`, `runs/<run>/samples/epoch_*.png` (progress over training) |

Tip: evaluate a few checkpoints (`epoch_004.pt`, `epoch_009.pt`, ...) — GANs don't always improve monotonically,
and an FID-vs-epoch table makes a nice addition to the report.
