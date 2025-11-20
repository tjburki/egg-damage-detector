import kagglehub
from torch import nn

# Download latest version
path = kagglehub.dataset_download("abdullahkhanuet22/eggs-images-classification-damaged-or-not")

print("Path to dataset files:", path)