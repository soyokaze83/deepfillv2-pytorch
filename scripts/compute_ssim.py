import argparse

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser(
    description="Compute SSIM metric between prediction and ground truth"
)
parser.add_argument("--pred", required=True, help="Path to prediction image")
parser.add_argument("--orig", required=True, help="Path to original/ground truth image")
parser.add_argument("--mask", required=True, help="Path to mask image")
args = parser.parse_args()

pred = np.array(Image.open(args.pred))
orig = np.array(Image.open(args.orig))
mask = np.array(Image.open(args.mask).convert("L")).astype(float) / 255.0

# Full image SSIM
if pred.ndim == 3:
    ssim_full = ssim(pred, orig, data_range=255, channel_axis=2)
else:
    ssim_full = ssim(pred, orig, data_range=255)

# Masked region SSIM (where mask > 0.5 = white = inpainted area)
# For masked SSIM, we compute SSIM map and average over masked region
mask_binary = mask > 0.5

if pred.ndim == 3:
    _, ssim_map = ssim(pred, orig, data_range=255, channel_axis=2, full=True)
    # Average SSIM map across channels, then apply mask
    ssim_map_avg = np.mean(ssim_map, axis=2)
else:
    _, ssim_map_avg = ssim(pred, orig, data_range=255, full=True)

ssim_masked = np.mean(ssim_map_avg[mask_binary])

print(f"full:{ssim_full:.6f}")
print(f"masked:{ssim_masked:.6f}")
