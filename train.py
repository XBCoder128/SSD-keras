from config import *
from model import get_SSD, MultiboxLoss
from tensorflow.keras.preprocessing import image
from utils import BBoxUtility, draw
from dataset import Generator
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle

priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASS, priors)

with open('train.keras', 'r') as f:
    gt = f.read()
    gt = gt.split('\n')

gen = Generator(gt, bbox_util, 3, (train_input_shape[0], train_input_shape[1]), do_crop=False)

model = get_SSD(train_input_shape, num_class=NUM_CLASS)

optim = tf.keras.optimizers.Adam(lr=base_lr)
model.compile(optimizer=optim, loss=MultiboxLoss(NUM_CLASS, neg_pos_ratio=2.0).compute_loss)
if os.path.exists('model.h5'):
    model.load_weights('model.h5')
print("TRUE")
nb_epoch = 100
try:
    history = model.fit_generator(gen.generate(True), gen.train_batches, nb_epoch)
except KeyboardInterrupt:
    print("训练中断")
model.save_weights('model.h5')
print('权重保存成功')