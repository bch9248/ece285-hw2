"""Run once on a node WITH internet (e.g. login node) so FID/IS work on offline
compute nodes. Weights are cached in $TORCH_HOME (default ~/.cache/torch)."""
from torchvision.models import Inception_V3_Weights, inception_v3

inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
print("InceptionV3 weights cached.")
