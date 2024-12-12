# Load model directly
from transformers import AutoImageProcessor, AutoModelForPreTraining
path = '/home/bao511/aaayanglib/domain adaptive/DODA-main/vit-mae-base'
processor = AutoImageProcessor.from_pretrained(path)
model = AutoModelForPreTraining.from_pretrained(path)
#python环境下
import torch
print(torch.__version__)