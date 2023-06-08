#imports
import torch
from super_gradients.training import models
import os
import cv2
import numpy as np
from tqdm import tqdm 
#paths
test_path = "./test/"
test_path_label = "./test_labels/"

#dataset parameters

dataset_params = {
    'test_images_dir':'./yolonas/test/',
    'test_labels_dir':'./yolonas/test_labels/',
    'classes': ['button']    
}

#load model
MODEL_ARCH = 'yolo_nas_l'
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MAX_EPOCHS = 5
CHECKPOINT_DIR = f'./training_backup/'
EXPERIMENT_NAME = f'yolo_nas_l_e5'

best_model = models.get(
    MODEL_ARCH,
    num_classes=len(dataset_params['classes']),
    checkpoint_path=f"{CHECKPOINT_DIR}/{EXPERIMENT_NAME}/average_model.pth"
).to(DEVICE)

image_path = './test/54_jpg.rf.94557887a7d02e755121438d9947f515.jpg'

best_model.predict(image_path).show()
