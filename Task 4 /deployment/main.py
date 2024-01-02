# Import necessary libraries
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from vit_keras import vit
from tensorflow.keras.layers import Layer


# Create a Flask web application instance
app = Flask(__name__)

# Load the pre-trained model for image classification
model = tf.keras.models.load_model('version2.h5')
# Define the target directory for saving uploaded images
target_img = os.path.join(os.getcwd(), 'static/images')

# Define the route for the home page
@app.route('/')
def index_view():
    return render_template('index.html')

# Allow files with extension png, jpg, and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXT

# Function to load and prepare the image in the right shape for model prediction
def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x 


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path) #prepressing method
            class_prediction=model.predict(img) 
            classes_x=np.argmax(class_prediction,axis=1)
            if classes_x == 0:
              state = "NORMAL"
            elif classes_x == 1:
              state = "COVID19"
            else:   
                state = "PNEUMONIA" 
                
      #'fruit' , 'prob' . 'user_image' these names we have seen in predict.html.
            return render_template('predict.html', state = state,prob=class_prediction, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"




# Define the route for image prediction

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
