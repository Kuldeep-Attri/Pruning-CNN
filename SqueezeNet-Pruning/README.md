# Pruning-CNN

This part is modified for "SqueezeNet" from the https://github.com/jacobgil/pytorch-pruning

At each pruning step 128 filters are removed from the network.

# Usage

This repository uses the PyTorch ImageFolder loader, so it assumes that the images are in a different directory for each category.

Train

......... dogs

......... cats

Test

......... dogs

......... cats


Training: python finetune.py --train

Pruning: python finetune.py --prune

Calculating Flops: python finetune.py --flops (change the model name in finetune.py)