# imports
from character_recognition import CharacterRecognizer
import io
from PIL import Image, ImageDraw, ImageFont
import PIL
import torch
from super_gradients.training import models
import os
import cv2
import numpy as np
from tqdm import tqdm
import imageio
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# paths
test_path = "./test/"
test_path_label = "./test_labels/"

# dataset parameters
dataset_params = {
    'test_images_dir': './yolonas/test/',
    'test_labels_dir': './yolonas/test_labels/',
    'classes': ['button']
}

# load model
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
# functions


def swap_columns(matrix):  # Yolonas swaps bounding box columns, hence fixing that
    for row in matrix:
        # Swap the 1st and 2nd columns
        row[0], row[1] = row[1], row[0]

        # Swap the 3rd and 4th columns
        row[2], row[3] = row[3], row[2]

    return matrix


# pre-processing function for the detected buttons
def button_candidates(boxes, scores, image):

    button_scores = []  # stores the score of each button (confidence)
    button_patches = []  # stores the cropped image that encloses the button
    button_positions = []  # stores the coordinates of the bounding box on buttons

    for box, score in zip(boxes, scores):
        if score < 0.5:
            continue

        y_min = int(box[0])
        x_min = int(box[1])
        y_max = int(box[2])
        x_max = int(box[3])

        if x_min < 0 or y_min < 0:
            continue
        button_patch = image[y_min: y_max, x_min: x_max]
        button_patch = cv2.resize(button_patch, (180, 180))

        button_scores.append(score)
        button_patches.append(button_patch)
        button_positions.append([x_min, y_min, x_max, y_max])
    return button_patches, button_positions, button_scores


############################################################################################
# Initializing an object for the characterRecognizer class
recognizer = CharacterRecognizer(verbose=False)
images_arr = []  # Array of image paths

# Generating a array with paths to test images
for file_name in os.listdir(test_path):
    images_arr.append(os.path.join(test_path, file_name))

times_det = []
times_lbl = []

for file_path in tqdm(images_arr):

    # Button detection
    with open(file_path, 'rb') as f:
        img_np = np.asarray(PIL.Image.open(io.BytesIO(f.read())))
    t0 = cv2.getTickCount()
    preds = best_model.predict(file_path)
    t1 = cv2.getTickCount()
    time = (t1-t0)/cv2.getTickFrequency()
    times_det.append(time)

    for button_pred in preds._images_prediction_lst:  # loops only once since only 1 image is passed at a time
        boxes = button_pred.prediction.bboxes_xyxy
        boxes = swap_columns(boxes)
        scores = button_pred.prediction.confidence

    button_patches, button_positions, _ = button_candidates(
        boxes, scores, img_np)

    # Button character recognition
    t0 = cv2.getTickCount()
    for button_img in button_patches:
        # get button text and button_score for each of the images in button_patches
        button_text, button_score, _ = recognizer.predict(button_img)
    t1 = cv2.getTickCount()
    time = (t1-t0)/cv2.getTickFrequency()
    times_lbl.append(time)

times_total = times_lbl + times_det  # Analyzing total times
arr = np.array(times_total)
# measures of dispersion
avg = np.round(np.mean(arr),5)
min = np.round(np.amin(arr),5)
max = np.round(np.amax(arr),5)
range = np.round(np.ptp(arr),5)
variance = np.round(np.var(arr),5)
sd = np.round(np.std(arr),5)


print("Average = ", avg)
print("Minimum =", min)
print("Maximum =", max)
print("Range =", range)
print("Variance =", variance)
print("Standard Deviation =", sd)
