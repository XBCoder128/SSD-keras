from config import *
from model import get_SSD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from utils import BBoxUtility, draw
import tensorflow as tf
import os

import matplotlib.pyplot as plt
import numpy as np
import pickle

priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASS, priors, nms_threshold=0.1)

inputs = []
images = []
img_path = 'test_pic/IMG_0703.JPG'
img = tf.io.read_file(img_path)
img = tf.image.decode_image(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)
w = img.shape[1]
h = img.shape[0]
minwh = min(w, h)
ratio = 300 / minwh
w *= ratio
h *= ratio
img = tf.image.resize(img, (300, 300))
img = tf.image.convert_image_dtype(img, tf.uint8)
images.append(plt.imread(img_path))
inputs = tf.expand_dims(img, axis=0)
inputs = preprocess_input(inputs)

model = get_SSD(img.shape, num_class=NUM_CLASS)  # 其实这里的尺寸可以根据输入图片来更改
if os.path.exists('model.h5'):
    model.load_weights('model.h5')

preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)
draw(images, results, NUM_CLASS, confidence=0.07)
