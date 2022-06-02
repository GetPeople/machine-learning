# -*- coding: utf-8 -*-
"""Copy of SiameseNetwork_v1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D5A5XiEWpmJZRly-agpVEE-6fRXt-Zy-

# Siamese Network
Network for a Face Recognition System

# Import Libraries
"""

!pip install keras_tuner
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K
import collections

from IPython.display import clear_output

#Tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Dense, Dropout, Flatten
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

#Scikitlearn
from sklearn.utils import shuffle

# Keras Tuner for Hyperparameter Tuning
import keras_tuner as kt

clear_output()

"""# Dataset

## Utility Functions
"""

## This function for making pairs  
## (both positive and negative pairs)
def generate_pairs(images, labels):
  # Generate label and index
  label_unique = np.unique(labels)                  # make a dictionary like this :
  label_indices = collections.defaultdict(list)     # label_indices = {
  i = 0                                             # 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  for idx, label in enumerate(labels):              # 1: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    if label == i:                                  # ...
      label_indices[label].append(idx)              # }
    else:
      i += 1

  # Generate image and labels pairs
  pair_images = []
  pair_labels = []
  for idx, image in enumerate(images):
    # positive pairs
    indices = label_indices.get(labels[idx])     # indices = [11, 12, 13, 14, 15, 16, 17, 18, 19]
    np.random.seed(42)
    positive = images[np.random.choice(indices)] # image = take random image from index above
    np.random.seed(42)
    pair_images.append((image, positive))        # make a positive pairs
    pair_labels.append(1)                        # (+) pairs labeled with 1

    # negative pairs
    indices = np.where(labels != labels[idx])         # neg_indices = other indices from indices above 
    negative = images[np.random.choice(indices[0])]   # neg_indices[0] -> take array from list
    pair_images.append((image, negative))             # make a negative pairs
    pair_labels.append(0)                             # (-) pairs labeled with 0
      
  return np.array(pair_images), np.array(pair_labels)

# Get the GDrive paths
# https://drive.google.com/file/d/1RWfKbQB-OMD8bIeyM0GnqovCA3VOWZgw/view?usp=sharing
!gdown 'https://drive.google.com/uc?id=1RWfKbQB-OMD8bIeyM0GnqovCA3VOWZgw'

# Unzip the file
!unzip /content/olivetti.zip
clear_output()

"""## Prepare Data"""

# Get the data directory paths
images_path = '/content/olivetti_faces.npy'
labels_path = '/content/olivetti_faces_target.npy'

# Load the images and labels
face_images = np.load(images_path)
face_labels = np.load(labels_path)

# Define the target shape
target_shape = face_images[0].shape

# Make a pairs of image, set label, and shuffle the dataset
images_dataset, labels_dataset = generate_pairs(face_images, face_labels)
images_dataset, labels_dataset = shuffle(images_dataset, labels_dataset)

"""## Visualize"""

def show_images(images, labels, cmap = 'gray', nrows = 10, pair = False):
  """
  images = ("face_images", "images_dataset")
  images must be an three dimensional array (n, x, y) or 
  four dimensional array (n, x, y, c) if there's 3 channel

  labels = ("face_labels", "labels_dataset")
  """
  # Count nrows
  ncols = nrows if pair else 10
  nrows = 2 if pair else nrows

  # Set the fig size
  x = ncols * 1.6
  y = nrows * 1.6
  figsize = (x, y)

  if pair:
    n = images.shape[0]
    p = images.shape[1]
    x, y = images.shape[2], images.shape[3]
    images = images.reshape(n*p, x, y)

  # Image
  fig,ax = plt.subplots(nrows, ncols,
                        figsize = figsize,
                        subplot_kw = dict(xticks=[], yticks=[]))  # remove x and y axis
  if pair:
    fig.suptitle('This is sample of {} pair faces.'.format(ncols), fontsize = 16)
    for col in range(ncols):
      for row in range(nrows):
        ax[row, col].imshow(images[col*nrows + row], cmap = cmap)   # print the images
        ax[1, col].set_xlabel(labels[col])                          # print the labels
  else:
    fig.suptitle('This is sample of {} faces of {} person.'.format(nrows*ncols, nrows),
                 fontsize = 16)
    for row in range(nrows):
      for col in range(ncols):
        ax[row, col].imshow(images[row*ncols + col], cmap = cmap)   # print the images
        ax[row, col].set_xlabel(labels[row*ncols + col])            # print the labels
  plt.show()

show_images(images = face_images, labels = face_labels, nrows = 4, pair = False)
show_images(images = images_dataset, labels = labels_dataset, nrows = 10, pair = True)

"""# Network Architecture

## Embedding Layer and Siamese Network
"""

# Define the Contrastive Loss
def contrastive_loss(y, preds, margin=1):
    y = tf.cast(y, preds.dtype)
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)

    return loss

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

def model_builder(hp):

  # Hyperparameter params
  hp_conv_filter_1 = hp.Choice('filter_1', values = [16, 32])
  hp_conv_filter_2 = hp.Choice('filter_2', values = [32, 64])
  hp_conv_filter_3 = hp.Choice('filter_3', values = [64, 128])
  hp_conv_filter_4 = hp.Choice('filter_4', values = [128, 256])
  hp_conv_kernel_size = hp.Choice('kernel', values = [3, 5, 7])
  hp_conv_activation = hp.Choice('activation', values = ["relu", "tanh"])
  hp_units_1 = hp.Choice('units_1', values = [2048, 4096])
  hp_units_2 = hp.Choice('units_2', values = [512, 1024])
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
 
  #Embedding layer
  embedding = Sequential([Conv2D(hp_conv_filter_1, (hp_conv_kernel_size,hp_conv_kernel_size), padding = "same", activation = hp_conv_activation, input_shape=(64, 64, 1)),
                          MaxPooling2D(pool_size=(2,2)),      
                          Dropout(0.3), 
                          Conv2D(hp_conv_filter_2, (hp_conv_kernel_size,hp_conv_kernel_size), padding = "same", activation = hp_conv_activation),
                          MaxPooling2D(pool_size=(2,2)),
                          Dropout(0.3),
                          Conv2D(hp_conv_filter_3, (hp_conv_kernel_size,hp_conv_kernel_size), padding = "same", activation = hp_conv_activation),
                          MaxPooling2D(pool_size=(2,2)), 
                          Dropout(0.3),
                          Conv2D(hp_conv_filter_4, (hp_conv_kernel_size,hp_conv_kernel_size), padding = "same", activation = hp_conv_activation),
                          Flatten(),
                          Dense(hp_units_1, activation = hp_conv_activation),
                          Dense(hp_units_2, activation = "sigmoid")
                          ])

  anchor_input = Input(name="anchor", shape=target_shape + (1,))
  compare_input = Input(name="compare", shape=target_shape + (1,))

  distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(compare_input),)
  
  outputs = layers.Dense(1, activation = "sigmoid") (distances)

  siamese_model = Model(inputs=[anchor_input, compare_input], 
                        outputs=outputs)
  
  
  siamese_model.compile(optimizer=Adam(hp_learning_rate),
                loss=BinaryCrossentropy(),
                metrics=['accuracy'])

  siamese_model.summary()
  
  return siamese_model

def training_model(x, y, 
                   model = model_builder, 
                   objective = 'val_accuracy', 
                   max_trials = 10, 
                   seed = 42, 
                   exe_per_trial = 1,
                   epochs = 20,
                   validation_split = 0.2,
                   batch_size = 64):
  
  tuner = kt.RandomSearch(model,
                          objective = objective,
                          max_trials = max_trials,
                          seed = seed,
                          executions_per_trial = exe_per_trial,
                          directory = '/tmp/tb')

  print("\n[INFO] This is a summary of possible parameters")
  tuner.search_space_summary()

  print("\n[INFO] Train model use hyperparameter tuning")
  tuner.search(x, y, epochs = epochs, validation_split = validation_split, batch_size = batch_size, callbacks=[tf.keras.callbacks.TensorBoard("/tmp/tb_logs")])

  best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
  print(f"""
  [INFO]
  The hyperparameter search is complete (yeayy!!). 
  The optimal number of :
  [1] filter in the 1st convolutional layer is {best_hps.get('filter_1')}, 
  [2] filter in the 2nd convolutional layer is {best_hps.get('filter_2')},
  [3] filter in the 3rd convolutional layer is {best_hps.get('filter_2')},
  [4] filter in the 4th convolutional layer is {best_hps.get('filter_4')},
  [5] kernel_size in all convolutional layer is {best_hps.get('kernel')},
  [6] activation function in all convolutional layer is {best_hps.get('activation')},
  [7] units number in the 1st fully connected layer is {best_hps.get('units_1')},
  [8] units number in the 2nd fully connected layer is {best_hps.get('units_2')},
  [9] learning rate for the optimizer is {best_hps.get('learning_rate')}.
  """)

  print("\n[INFO] This is model summary from top best 5 parameter combination")
  tuner.results_summary(num_trials = 5)

  return tuner, best_hps

"""# Training"""

x = [images_dataset[:, 0, :], images_dataset[:, 1, :]]
y = labels_dataset

tuner, best_hps = training_model(x, y)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# %tensorboard --logdir /tmp/tb_logs

# Re run best model in training data
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(x, y, epochs=100, validation_split=0.2)

#Visualizing the loss and accuracy (masih error)
'''
visualize = (plt.plot(hypermodel.history["loss"]),
             plt.plot(hypermodel.history["val_loss"]),
             plt.plot(hypermodel.history["accuracy"]),
             plt.plot(hypermodel.history["val_accuracy"]),
             plt.legend(["Loss", "Validation Loss", "Accuracy", "Validation Accuracy"]))
'''

hypermodel.save('/content/siamese_model')

def siamese_model(x, y, batch_size=64, dropout=0.3, metrics='accuracy', epochs=150, pool_size=2, validation_split=0.2, summary=False, callbacks=None, checkpoint=None):
 
  #Embedding layer
  embedding = Sequential([Conv2D(32, (5,5), padding="same", activation="relu",input_shape=(64, 64, 1)),
                      MaxPooling2D(pool_size=(pool_size ,pool_size)),      
                      Dropout(dropout), 
                      Conv2D(64, (5,5), padding="same", activation="relu"),
                      MaxPooling2D(pool_size=(pool_size ,pool_size)),
                      Dropout(dropout),
                      Conv2D(64, (5,5), padding="same", activation="relu"),
                      MaxPooling2D(pool_size=(pool_size ,pool_size)), 
                      Dropout(dropout),
                      Conv2D(256, (5,5), padding="same", activation="relu"),
                      Flatten(),
                      Dense(4096, activation = "relu"),
                      Dense(1024, activation = "sigmoid")
  ])

  anchor_input = Input(name="anchor", shape=target_shape + (1,))
  compare_input = Input(name="compare", shape=target_shape + (1,))

  distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(compare_input),)
  
  outputs = layers.Dense(1, activation = "sigmoid") (distances)

  siamese_model = Model(inputs=[anchor_input, compare_input], 
                        outputs=outputs)
  
  
  siamese_model.compile(optimizer=Adam(), 
                loss=BinaryCrossentropy(), 
                metrics=[metrics])

  if summary:
    siamese_model.summary()
  
  history = siamese_model.fit(x=x, 
                      y=y,
                      epochs=150, 
                      validation_split = validation_split, 
                      batch_size = batch_size,
                      callbacks = [callbacks])
  
  #Visualizing the loss and accuracy
  visualize = (plt.plot(history.history["loss"]),
  plt.plot(history.history["val_loss"]),
  plt.plot(history.history["accuracy"]),
  plt.plot(history.history["val_accuracy"]),
  plt.legend(["Loss", "Validation Loss", "Accuracy", "Validation Accuracy"]))

  
  return history, siamese_model, visualize

callbacks = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
history, siamese_model, visualize = siamese_model(x=[images_dataset[:, 0, :], images_dataset[:, 1, :]],
                                                  y=labels_dataset, epochs=150, validation_split = 0.2, batch_size = 64,
                                                  callbacks = [callbacks])

"""# Testing
Hasil testing aneh, mungkin bisa diambil weight saja dari best model dan di run kembali di tensorflow biasa.
"""

def visualize_images(images, n = 5):
    #Visualize the images

    def visual(ax, image):
        ax.imshow(image, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9)) 
    axs = fig.subplots(1, n)
    for i in range(n):
        visual(axs[i], images[i])

def predict(image_pairs):
  scores = []
  score = siamese_model.predict([image_pairs[:, 0, :], image_pairs[:, 1, :]])
  scores.append(score)
  print(scores)
  return scores

def pairs_test(i,n):
  #Visualizes the images and the prediction and show the prediction results
  test_index = random.sample(range(i), 1)[0]
  test_image = face_images[test_index]
  fig,ax = plt.subplots(figsize = (3,3),
                        subplot_kw = dict(xticks=[], yticks=[])) 
  ax.imshow(test_image, cmap='gray')
  compare_images = []
  for i in range(n):
      index = random.sample(range(i * 10, (i + 1) * 10), 1)[0]
      images = face_images[index]
      compare_images.append(images)
  visualize_images(compare_images)
  image_pairs = []

  for images in compare_images:
      image_pairs.append((test_image, images))
    
  image_pairs = np.array(image_pairs)
  return image_pairs

pred = predict(pairs_test(50,5))
result = [pred[0][i][0] for i in range(5)]
result = np.array(result)
print(result.argsort() + 1)

"""# Testing Manual"""

!unzip "/content/database_wajah_artis.zip" -d "/content/database_wajah_artis/"
clear_output()

base = '/content/database_wajah_artis/database_wajah/database'
dirs = os.listdir(base)
paths = []

for dir in dirs:
  path = os.path.join(base, dir)
  paths.append(path)

images_dataset[0][0]

import cv2
from PIL import Image

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

images = import_images(paths)

images[0].shape

test_image_path_1 = ['/content/database_wajah_artis/database_wajah/test/test_1.png']
test_image_path_2 = ['/content/database_wajah_artis/database_wajah/test/test_2.png']
test_image_path_3 = ['/content/test_4.png']

test_image_1 = import_images(test_image_path_1)
test_image_2 = import_images(test_image_path_2)
test_image_3 = import_images(test_image_path_3)

def pairing_image(test, database):
  image_pair = []
  for i in range(10):
    image_pair.append((test[0], database[i]))
  return image_pair

def visualize_images(images, n = 5, id = range(9)):
    #Visualize the images
    def visual(ax, image):
        ax.imshow(image, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(20, 20)) 
    axs = fig.subplots(1, n)
    for i, x in enumerate(id):
        visual(axs[i], images[x])

def predict(pair):
  scores = []
  score = siamese_model.predict([pair[:, 0, :], pair[:, 1, :]])
  scores.append(score)
  return scores

paired_images = np.array(pairing_image(test_image_3, images))

pred = predict(paired_images)
result = [pred[0][i][0] for i in range(10)]
ranking = np.array(result).argsort().tolist()

id = {}
for i, x in enumerate(ranking):
  id[x] = i

sort_keys = id.items()
new_id = sorted(sort_keys, reverse = True)

sorted_index = [x for i, x in new_id]

plt.imshow(test_image_3[0], cmap = "gray")

visualize_images(images, n = 9, id = range(9))

print('Top 5 Most Similar Face :')
visualize_images(images, n = 5, id = sorted_index[:5])

"""# Reference
Hyperparameter Tuning :
- https://www.tensorflow.org/tutorials/keras/keras_tuner
- https://medium.com/swlh/hyperparameter-tuning-in-keras-tensorflow-2-with-keras-tuner-randomsearch-hyperband-3e212647778f
"""

