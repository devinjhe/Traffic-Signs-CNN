import numpy as np
import sys
import tensorflow as tf
import os
import cv2

# Check command-line arguments
if len(sys.argv) != 3:
    sys.exit("Usage: python whatSign.py <model> <image>")
model = tf.keras.models.load_model(sys.argv[1])
img = cv2.imread(sys.argv[2])
res = cv2.resize(img, (30, 30), interpolation=cv2.INTER_AREA)

classification = model.predict(
    [np.array(res).reshape(1, 30, 30, 3)]
).argmax()

print(classification)
