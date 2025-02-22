"""
A simple experiment about what happens without Direct Ascent Synthesis (DAS).
Check the DAS paper: https://arxiv.org/abs/2502.07753

This *very* simple approach is for comparing to DAS.
"""
import warnings
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import torch
import clip
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse
from cliptools import fix_random_seed, raw_to_real_image, real_to_raw_image

# Argument Parsing
parser = argparse.ArgumentParser(description='CLIP gradient ascent')
parser.add_argument('--model_name', default='ViT-B/32')
parser.add_argument('--use_image', type=str, default="images/cat.png", help="Path to a single image")
parser.add_argument("--results_dir", type=str, default="PGD-attack", help="Base directory to save all outputs to, default: 'PGD-attack'")
# Attack (optimization) settings:
parser.add_argument('--eps', default=0.1, type=float, help="Maximum Perturbation epsilon; default: 0.1")
parser.add_argument('--lr', default=0.2, type=float, help="Step Size alpha (= Learning Rate); default: 0.2")
parser.add_argument('--steps', default=100, type=int, help="Total Attack steps; default: 100")
# Make it deliberately adversarial (misleading), or try to optimize towards embedding
parser.add_argument('--adv', action='store_true', help="Adversarial Attack (min cosine similarity). If not --adv, optimizes towards max cos sim")
# For reproducable results:
parser.add_argument("--deterministic", action='store_true', help="Use deterministic behavior (CUDA backends, torch, numpy)")

args = parser.parse_args()

if args.adv:
    meth = "adv"
else:
    meth = "norm"

if args.deterministic:
    fix_random_seed()
    det = "det"
else:
    det = "rnd"

results_dir = args.results_dir
steps = args.steps

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(args.model_name, device=device)

image_path = args.use_image
original_image = Image.open(image_path).convert("RGB")
original_tensor = preprocess(original_image).unsqueeze(0).to(device)


# Attack Parameters
epsilon = args.eps  # Maximum perturbation
alpha = args.lr   # Step size / LR
num_steps = args.steps # Number of attack steps

gradient_sign_only=False # Wasn't really needed with this, but set to True to experiment

jitter_scale=56
noise_scale=0.2

# Initialize perturbation (direct zero tensor)
perturbation = torch.zeros_like(original_tensor, requires_grad=True)
optimizer = torch.optim.Adam([perturbation], lr=alpha)

# Get original CLIP embedding
with torch.no_grad():
    original_embedding = model.encode_image(original_tensor)

# Optimization loop
for step in range(num_steps):
    optimizer.zero_grad()

    adversarial_image = raw_to_real_image(original_tensor + perturbation)
    adversarial_embedding = model.encode_image(adversarial_image)

    if args.adv:
        loss = torch.cosine_similarity(adversarial_embedding, original_embedding).mean()
        (-loss).backward()
    else:
        loss = torch.cosine_similarity(adversarial_embedding, original_embedding).mean()
        loss.backward()

    if gradient_sign_only:
        perturbation.grad.data.sign_()

    optimizer.step()

    perturbation.data = torch.clamp(perturbation, -epsilon, epsilon)

    if step % 10 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

# Convert adversarial image back to displayable format for saving
adv_image_np = raw_to_real_image(original_tensor + perturbation).detach().cpu().squeeze().permute(1, 2, 0).numpy()
perturbation_np = perturbation.detach().cpu().squeeze().permute(1, 2, 0).numpy()

# Enhance visual perception for humans
def unsharp_mask(image, sigma=1.0, strength=2.0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return np.clip(image + strength * (image - blurred), 0, 1)

def reduce_contrast(image, factor=0.85):
    return np.clip((1 - factor) * image + factor * 0.5, 0, 1)

adv_amplified = unsharp_mask(adv_image_np, sigma=1.0, strength=5.0)
adv_amplified = reduce_contrast(adv_amplified, factor=0.2)

adv_image_np = reduce_contrast(adv_image_np, factor=0.2)

fig, ax = plt.subplots(1, 4, figsize=(12,4))
ax[0].imshow(original_image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(adv_image_np)
ax[1].set_title("Adversarial (Attack)")
ax[1].axis("off")

ax[2].imshow(adv_amplified)
ax[2].set_title("Adversarial (Amplified)")
ax[2].axis("off")

ax[3].imshow(perturbation_np * 5 + 0.5, cmap="gray")
ax[3].set_title("Perturbation (Amplified)")
ax[3].axis("off")

img_name = os.path.splitext(os.path.basename(args.use_image))[0]
os.makedirs(args.results_dir, exist_ok=True)

# Regex to match filenames of the format, prevent overwriting
pattern = re.compile(rf"{img_name}_{meth}_{epsilon}_{alpha}_{steps}_{det}_(\d+)\.png")

existing_numbers = []
for filename in os.listdir(results_dir):
    match = pattern.match(filename)
    if match:
        existing_numbers.append(int(match.group(1)))

next_num = max(existing_numbers, default=-1) + 1

# Ensure atomicity by looping until a truly unique filename is found
while True:
    combined_filename = f"{img_name}_{meth}_{epsilon}_{alpha}_{steps}_{det}_{next_num}.png"
    combined_path = os.path.join(results_dir, combined_filename)
    if not os.path.exists(combined_path):
        break
    next_num += 1  # Increment in case of rare race conditions

plt.savefig(combined_path, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)

print(f"Image saved to: {combined_path}")