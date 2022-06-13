# Machine Learning

## Tools :
  - Google Colab (due limited resources of GPU)
  - Visual Studio Code

## Make a model :
### 1. Preparation
  - Load the datasets
    - Using olivetti face dataset from AT&T Laboratories Cambridge for training<br>(here's the link to dataset: https://drive.google.com/file/d/14DwboVbHw042bvZSiM7HwBnscamv44pt/view?usp=sharing)
    - Using database_wajah as our victims simulation database <br>(here's the link to database : https://github.com/GetPeople/machine-learning/tree/main/database_wajah)
  - Convert image to :
    - Grayscale
    - Cropped only at the face (using prebuilt model from MTCNN)
    - Resize 64*64 pixels
  - Pair the image and set the label (1 for same person and 0 for different person)
    - For every face person image paired with positive (from same person) and negative (from different person)
### 2. Build the architecture
  - Using CNN for embedding each picture
    - Add `Conv2D` layer
    - Add `MaxPooling` layer
    - Add `Dropout` layer
    - Repeat all above 3 times
    - Add `Conv2D` layer
    - Add `Flatten` layer
    - Add `Dense` layer
    - Add `Dense` layer with **sigmoid** activation function
  - Take single value from each embedding and measure the euclid distance
  - Take the distance as input for sigmoid function
### 3. Hyperparameter Tuning
  - Using random search from `keras_tuner` to find optimal value of :
    - number of filter in each convolutional layer
    - kernel size for all convolutional layers
    - activation function for all convolutional layers
    - unit number in each fully connected layer
    - learning rate for the optimizer
  - Retrain the model using best hyperparameters and got the result : <br>
  ![Result](https://drive.google.com/uc?export=view&id=1QSNNZcIKau4FTWqjyZLauEL3qeN8_nCc)
    - `loss : 29.70%`
    - `accuracy : 99.06%`
    - `val_loss : 32.17%`
    - `val_accuracy : 98.12%` 
### 4. Save model
  - Save best model weight .h5 use `model.save` from tensorflow to google drive <br>(here's the link : https://drive.google.com/file/d/1ZntMb0khSnUU-Cozq2okpkHtYI4XOi9F/view?usp=sharing)
