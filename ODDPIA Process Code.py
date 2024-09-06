import matplotlib.pyplot as plt import numpy as np
import os import PIL
import tensorflow as tf import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential img_height = 180
img_width = 180

#print ("File %s" % (sys.argv[1]))

model = keras.models.load_model('D:\\Kuyabenj\\Kuya\\code\\eye-disease') class_names = ['CATARACT', 'CONJUNCTIVITIS', 'GLAUCOMA', 'HEALTHY EYES', 'KERATITIS', 'UVEITIS']
##
##sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/cat1.png"  ##sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = keras.preprocessing.image.load_img( sys.argv[1], target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img) img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0]) print(
"result:{}:{:.2f}:"
.format(class_names[np.argmax(score)], 100 * np.max(score))
)
