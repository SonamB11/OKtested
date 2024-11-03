from flask import Flask, render_template, request
import easyocr
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved product classifier model
model = load_model('product_classifier_model.h5')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Define class labels for product detection
class_labels = ['act2', 'biotique', 'blackgola', 'boost', 'bounty', 'bournville', 
                'bournvita', 'chanaflour', 'cocacola', 'colgate', 'dairymilk', 
                'doritos', 'fanta', 'ferrero', 'fizz', 'galaxy', 'garnier', 
                'greenmoongdal', 'groundnut', 'gullu', 'haldiram', 'heinz', 
                'hersheys', 'hidenseek', 'horlicks', 'idly_ravva', 'kissanjam', 
                'kissanketchup', 'kisses', 'kitkat', 'krishna_salt', 'lays', 
                'lindt', 'maaza', 'maida', 'mars', 'minutemaid', 'mountain_dew', 
                'oats', 'oreo', 'paper_boat', 'parleg', 'passion_indulge', 
                'perk', 'pipo', 'predatorenergy', 'pringles', 'protinex', 
                'raavi_sauce', 'rajma', 'redbull', 'redhunt', 'redwine', 'revlon', 
                'rimzim', 'similac', 'snickers', 'sprite', 'sting', 'sugar', 
                'thumsup', 'tonicwater', 'toordal', 'tooyum', 'tropicana', 'twix', 
                'unibic', 'vcare', 'vlcc', 'wingreens']

# Function to run OCR
def run_ocr(img_path):
    image = cv2.imread(img_path)
    result = reader.readtext(image)
    return result

# Function to run product classification
def classify_product(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img) / 255.0  # Rescale pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return class_labels[predicted_class]

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        # Use .get() to avoid KeyError
        task = request.form.get('task')
        file = request.files.get('image')

        if file and task:
            img_path = 'static/uploads/' + file.filename
            file.save(img_path)
            
            if task == 'OCR':
                ocr_result = run_ocr(img_path)
                return render_template('process.html', task='OCR', img_path=img_path, ocr_result=ocr_result)
            elif task == 'Image Recognition':
                product = classify_product(img_path)
                return render_template('process.html', task='Image Recognition', img_path=img_path, product=product)
    
    return render_template('process.html')

if __name__ == '__main__':
    app.run(debug=True)
