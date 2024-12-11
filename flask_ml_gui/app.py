from flask import Flask, render_template, request
import numpy as np
import pickle
#import your_ml_model  # Replace with your machine learning model import

app = Flask(__name__)

# Load your pre-trained machine learning model here
dnnmodel = pickle.load(open('models/dnn.pkl', 'rb'))  
gbmodel = pickle.load(open('models/gbmodel.pkl', 'rb'))
sgdmodel = pickle.load(open('models/sgdmodel.pkl', 'rb'))  
xgbmodel = pickle.load(open('models/xgbmodel.pkl', 'rb'))  



@app.route("/")
def index():
    select_options = {
        'cap-shape': ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken', 'oval'],
        'cap-color': ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow', 'black'],
        'does-bruise-or-bleed': ['yes', 'no'],
        'veil-type': ['partial', 'universal'],
        'veil-color': ['brown', 'orange', 'white', 'yellow', 'black'],
        'ring-type': ['cobwebby', 'evanescent', 'flaring', 'large', 'none', 'pendant', 'sheathing', 'zone'],
        'spore-print-color': ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'],
        'habitat': ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'],
        'season': ['autumn', 'spring', 'summer', 'winter'],
        'stem-color': ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'],
        'stem-surface':['fibrous', 'scaly', 'silky', 'smooth', 'grooved'],
        'gill-attachment': ['attatched', 'descending', 'free', 'notched'],
        'gill-spacing': ['close', 'crowded', 'distant'],
        'cap-surface':['fibrous', 'grooves', 'scaly', 'smooth', 'silky'],
        'gill-color': ['black','brown','buff','chocolate','gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'],
        'stem-root': ['bulbous', 'club', 'cup', 'equal', 'rhizomorphs', 'rooted', 'scurfy', 'fibrous'],
        'has-ring': ['yes', 'no']
    }
    return render_template("index.html", select_options = select_options)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input from the form
        feature_names = ['cap-diameter', 'cap-shape', 'cap-surface','cap-color', 'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color', 'stem-height', 'stem-width', 'stem-root', 'stem-surface', 'stem-color','veil-type', 'veil-color', 'has-ring', 'ring-type', 'spore-print-color', 'habitat', 'season']
        features = np.array([0.0])
        for name in feature_names:
            features = np.append(features, float(request.form.get(name)))
        # Preprocess the input data for your model (if needed)
        # ... your data preprocessing code here ...

        # Make prediction using your model
        model = request.form.get('model')
        if model == 'dnn':
            prediction = dnnmodel.predict(features.reshape(1, -1)) > 0.5  # Assuming a list input
        elif model == 'gb':
            prediction = gbmodel.predict(features.reshape(1, -1)) > 0.5
        elif model == 'sgd':
            prediction = sgdmodel.predict(features.reshape(1, -1)) > 0.5
        elif model == 'xgb':
            prediction = xgbmodel.predict(features.reshape(1, -1)) > 0.5

        # Format the prediction for display
        predicted_class = prediction[0]  # Assuming single class output

        return render_template("result.html", prediction=predicted_class)

    else:
        return "Something went wrong. Please try again."

if __name__ == "__main__":
    app.run(debug=True)