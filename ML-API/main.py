# Util.py
import util
from util import convert_database, convert_image, take_image, pairing_image, predict, sort_path, predict

#Tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Dense, Dropout, Flatten
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

#Flask
from flask import Flask, request, jsonify, render_template

# Import saved model
model = load_model('my_h5_model.h5', custom_objects={'DistanceLayer' : util.DistanceLayer})

# Initialize Flask server with error handling
app = Flask(__name__)

@app.route('/refresh', methods = ["POST"])
def reload(): 
    convert_database()
    return render_template('upload.html')

@app.route('/', methods = ["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template('upload.html')
    elif request.method == "POST":
        image = request.files.get('file')
        try:
            image_path = "F:/ml-api/images/" + image.filename
            image.save(image_path)

            # Transform test image from request
            test_image = convert_image([image_path])
            
            # Ini diubah sesuai alamat database
            database, paths = take_image('F:/ml-api/converted/')

            # Make a pair
            n_images = len(database) # Untuk menghitung banyaknya images di database
            paired_images = pairing_image(test_image, database, n_images)

            # Prediction
            path_to_csv = "F:\ml-api\identitas.csv"
            pred = predict(model, paired_images, n_images, paths, path_to_csv)
            return pred
        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, use_reloader = False, port = 5000)