import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INI AKU GATAU BUAT APA

import numpy as np
import os
import tensorflow.keras.backend as K
import cv2

from PIL import Image

#Tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Dense, Dropout, Flatten
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

#Flask
from flask import Flask, request, jsonify, render_template

# Distance Layer
class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance
    between the embeddings
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, compare):
        sum_squared = K.sum(K.square(anchor - compare), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_squared, K.epsilon()))

# Import saved model
model = load_model('my_h5_model.h5', custom_objects={'DistanceLayer' : DistanceLayer})

# Convert image to array
def import_images(paths):
    images = []
    for path in paths:
      image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
      # rgb_weights = [0.2989, 0.5870, 0.1140]
      # image = np.dot(image, rgb_weights)
      image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
      image = Image.fromarray(image)
      image = image.resize((64,64))
      image = np.asarray(image) / 255
      images.append(image)
    return images

# Convert image to array
def transform_image(file):
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    # rgb_weights = [0.2989, 0.5870, 0.1140]
    # image = np.dot(image, rgb_weights)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    image = Image.fromarray(image)
    image = image.resize((64,64))
    image = np.asarray(image) / 255
    return image

# Pairing Image
def pairing_image(test, database):
    image_pair = []
    for i in range(10):
      image_pair.append((test[0], database[i]))
    return image_pair

# Predict result
def predict(pair):
    scores = []
    score = model.predict([pair[:, 0, :], pair[:, 1, :]])
    scores.append(score)
    return scores

# Initialize Flask server with error handling
app = Flask(__name__)

@app.route('/', methods = ["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('upload.html')
    elif request.method == "POST":
        image = request.files.get('file')
        try:
            image_path = "F:/coba_api/images/" + image.filename
            image.save(image_path)

            # Transform test image from request
            test_image = import_images([image_path])

            base = 'F:/coba_api/database_wajah/database/' # INI DIUBAH SESUAI DIRECTORY LOCAL (DATABASE ONLY)
            dirs = os.listdir(base)
            paths = []

            for dir in dirs:
                path = os.path.join(base, dir)
                paths.append(path)

            # Import and Transform Images (Database Only)
            images = import_images(paths)

            # Make a pair
            paired_images = np.array(pairing_image(test_image, images))

            # Prediction
            pred = predict(paired_images)
            result = [pred[0][i][0] for i in range(10)]
            dict_from_list = dict(zip(result, paths))
                        
            sort_dictionary= dict(sorted(dict_from_list.items(), key=lambda item: item[0], reverse = True)) 
            sorted_path = [values for key, values in sort_dictionary.items()]

            # Convert to JSON
            result = json.dumps(sorted_path[:5])
            return result
        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, use_reloader = False)