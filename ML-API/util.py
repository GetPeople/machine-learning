import os
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
import cv2
import json

from mtcnn.mtcnn import MTCNN

from PIL import Image

from tensorflow.keras import layers

# Distance Layer
class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, compare):
        sum_squared = K.sum(K.square(anchor - compare), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_squared, K.epsilon()))

# Cropping Image
def cropping_image(test_image):
    crop = MTCNN()
    results = crop.detect_faces(test_image)

    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height

    image = test_image[y1:y2, x1:x2]

    image = Image.fromarray(image)
    image = image.resize((64, 64))
    image = np.asarray(image)
    return image

# Import database
def generate_paths(base):
    dirs = os.listdir(base)
    paths = []

    for dir in dirs:
        path = os.path.join(base, dir)
        paths.append(path)
    return paths

def convert_database():
    base = 'F:/ml-api/database_wajah/'
    paths = generate_paths(base)
    images = crop_edit_image(paths)
    for image,path in zip(images,paths):
        convert_path = "F:/ml-api/converted/" + path.split("/")[-1]
        image = image * 255
        image = Image.fromarray(image)
        image = image.convert('L')
        image.save(convert_path)

# Take converted image in database
def take_image(base):
    paths = generate_paths(base)
    images = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = image / 255
        images.append(image)
    return images, paths

# Take image from the path, convert to array, 
# crop the image, convert to grayscale, resize, and 
# convert to array
def crop_edit_image(paths):
    images = []
    for path in paths:
      image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
      image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
      image = cropping_image(image)
      image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
      image = Image.fromarray(image)
      image = image.resize((64,64))
      image = np.asarray(image) / 255
      images.append(image)
    return images

# Save the image after crop edit to converted folder
def convert_image(path):
    image = crop_edit_image(path)[0]
    convert_path = "F:/ml-api/images/" + path[0].split("/")[-1]
    pic = image * 255
    pic = Image.fromarray(pic)
    pic = pic.convert('L')
    pic.save(convert_path)
    return image

# Pairing Image
def pairing_image(test, database, n_images):
    image_pair = []
    for i in range(n_images):
      image_pair.append((test, database[i]))
    return np.array(image_pair)

# Predict result
def sort_path(model, pair, n_images, paths_to_database):
    scores = []
    score = model.predict([pair[:, 0, :], pair[:, 1, :]])
    scores.append(score)
    result = [scores[0][i][0] for i in range(5)]
    dict_from_list = dict(zip(result, paths_to_database))
                
    sort_dictionary = dict(sorted(dict_from_list.items(), key=lambda item: item[0], reverse = True)) 
    sorted_path = [values for key, values in sort_dictionary.items()]
    return sorted_path

def predict(model, pair, n_images, paths_to_database, path_to_csv):
    information = []
    paths = sort_path(model, pair, n_images, paths_to_database)
    img_id = [path.split("/")[-1].split(".")[0] for path in paths]
    identitas = pd.read_csv(path_to_csv)
    identitas = identitas.to_dict()
    for index, path in zip(img_id, paths):
        index = int(index)
        information.append({
            'id' : index,
            'name' : identitas['nama_lengkap'][index-1], 
            'gender' : identitas['jenis_kelamin'][index-1],
            'birthPlace' : identitas['tempat_lahir'][index-1], 
            'birthDate' : identitas['tanggal_lahir'][index-1], 
            'posko' : identitas['lokasi_pengungsian'][index-1], 
            'nik' : identitas['nik'][index-1],
            'momName' : identitas['nama_ibu_kandung'][index-1],
            'photoUrl' : path
        })
    return json.dumps(information)