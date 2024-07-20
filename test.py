from ultralytics import YOLO
import cv2
import cvzone
import math
import torch

# Kiểm tra xem CUDA có khả dụng không
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')