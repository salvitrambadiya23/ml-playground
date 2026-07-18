# Flower Classification

CNN-based image classifier using PyTorch and transfer learning (ResNet18) on the Flowers Recognition dataset (5 classes: daisy, dandelion, rose, sunflower, tulip).

## Results
- Validation Accuracy: 92%
- See confusion_matrix.png for per-class breakdown

## Files
- `train.py` — trains the model, saves `flower_model.pth`
- `evaluate.py` — generates classification report + confusion matrix
- `predict.py` — predicts a single image: `py predict.py <path>`

## Usage
py train.py
py evaluate.py
py predict.py data/flowers/rose/example.jpg