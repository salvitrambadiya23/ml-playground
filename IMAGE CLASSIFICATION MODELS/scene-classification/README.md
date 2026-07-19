# Scene Classification (Intel Image Dataset)

CNN-based image classifier using PyTorch and transfer learning (ResNet18), trained on Kaggle with GPU acceleration. Classifies natural scenes into 6 categories: buildings, forest, glacier, mountain, sea, street.

## Dataset
Intel Image Classification dataset (puneet6060) — ~14,000 training images, 3,000 test images.

## Approach
- Transfer learning with pretrained ResNet18
- Data augmentation (random flip, rotation, color jitter) to reduce overfitting
- Learning rate scheduling (halved every 5 epochs)
- Checkpointing — only the best-performing epoch (by test accuracy) is saved

## Results
- Initial model (no augmentation): 93.97% train / 82.33% test accuracy (significant overfitting)
- Improved model (with augmentation): 92.28% train / 90.90% test accuracy
- Final model (augmentation + LR scheduling + checkpointing): best test accuracy captured automatically during training

## Files
- `image-classification-model.ipynb` — full training notebook (run on Kaggle with GPU)
- `scene_model_best.pth` — trained model weights (not included in repo, see .gitignore)

## Notes
Trained entirely on Kaggle Notebooks using free GPU (T4), rather than locally.