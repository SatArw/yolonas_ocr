#imports
import torch
from super_gradients.training import models
import os
import cv2
import numpy as np
from tqdm import tqdm
import imageio
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import PIL
from PIL import Image, ImageDraw, ImageFont
import io
from character_recognition import CharacterRecognizer

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

########################################################################################
#functions
def swap_columns(matrix):
    for row in matrix:
        # Swap the 1st and 2nd columns
        row[0], row[1] = row[1], row[0]
        
        # Swap the 3rd and 4th columns
        row[2], row[3] = row[3], row[2]
    
    return matrix

def button_candidates(boxes, scores, image): 
    button_scores = [] #stores the score of each button (confidence)
    button_patches = [] #stores the cropped image that encloses the button
    button_positions = [] #stores the coordinates of the bounding box on buttons

    for box, score in zip(boxes, scores):
        if score < 0.5: continue

        y_min = int(box[0])
        x_min = int(box[1])
        y_max = int(box[2])
        x_max = int(box[3])
        
        if x_min <0 or y_min < 0: continue
        button_patch = image[y_min: y_max, x_min: x_max]
        if not button_patch.any():
            print(x_min,y_min,x_max,y_max)
        button_patch = cv2.resize(button_patch, (180, 180))

        button_scores.append(score)
        button_patches.append(button_patch)
        button_positions.append([x_min, y_min, x_max, y_max])
    return button_patches, button_positions, button_scores

############################################################################################
recognizer = CharacterRecognizer(verbose=False)
images_arr = [] #Array of image paths

for file_name in os.listdir(test_path):
    images_arr.append(os.path.join(test_path, file_name))

times_det = []
times_lbl = []
for file_path in (images_arr[10:20]):
    #button detection
    with open(file_path, 'rb') as f:
        img_np = np.asarray(PIL.Image.open(io.BytesIO(f.read())))
        
    img_show = np.copy(img_np) #Copy of image, results will be displayed using this
    t0 = cv2.getTickCount()
    preds = best_model.predict(file_path)
    t1 = cv2.getTickCount()
    time = (t1-t0)/cv2.getTickFrequency()
    times_det.append(time)

    for button_pred in preds._images_prediction_lst: #loops only once since only 1 image is passed at a time
        boxes = button_pred.prediction.bboxes_xyxy
        boxes = swap_columns(boxes)
        scores = button_pred.prediction.confidence
        
    button_patches, button_positions, _ = button_candidates(boxes, scores, img_np)
    
    #button character recognition
    t0 = cv2.getTickCount()
    for button_img, button_pos in zip(button_patches, button_positions):
        button_text, button_score, button_draw = recognizer.predict(button_img, draw = True)
        xmin, ymin, xmax, ymax = button_pos
        button_rec = cv2.resize(button_draw,(xmax-xmin, ymax-ymin))
        img_show[ymin+6:ymax-6, xmin+6:xmax-6] = button_rec[6:-6,6:-6] #Placing the detection results on relevant areas of the image
    t1 = cv2.getTickCount()
    time = (t1-t0)/cv2.getTickFrequency()
    times_lbl.append(time)
    
    cv2.imshow('panels',img_show)
    cv2.waitKey(0)
    


  