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

images_arr = [] #Array of image paths

for file_name in os.listdir(test_path):
  images_arr.append(os.path.join(test_path, file_name))

times = []
for file_path in tqdm(images_arr):
  t0 = cv2.getTickCount()
  x = best_model.predict(file_path)
  t1 = cv2.getTickCount()
  time = (t1-t0)/cv2.getTickFrequency()
  times.append(time)

# print(mean(times))
arr = np.array(times)

# measures of dispersion
avg = np.mean(arr)
min = np.amin(arr)
max = np.amax(arr)
range = np.ptp(arr)
variance = np.var(arr)
sd = np.std(arr)
 
# print("Array =", arr)
print("Average = ", avg)
print("Minimum =", min)
print("Maximum =", max)
print("Range =", range)
print("Variance =", variance)
print("Standard Deviation =", sd)