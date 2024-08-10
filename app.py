from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import pickle
import shap


app = Flask(__name__)

filepath = "Model.pkl"
model = pickle.load(open(filepath, 'rb'))
explainer = shap.Explainer(model)

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():  # route which will take you to the prediction page
    return render_template('indexnew.html')

@app.route('/Home', methods=['POST', 'GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict', methods=["POST", "GET"])  # route to show the predictions in a web UI
def predict():
    if request.method == "POST":
        input_features = [float(x) for x in request.form.values()]
        features_values = [np.array(input_features)]

        feature_name=['Make', 'Vehicle_class', 'Engine_size', 'Cylinders',
       'Transmission', 'Fuel_type', 'Fuel_consumption_city(L/100 km)',
       'Fuel_consumption_hwy(L/100 km)', 'Fuel_consumption_comb(mpg)']

        x = pd.DataFrame(features_values, columns=feature_name)
        prediction = model.predict(x)
        numeric_features = ['Engine_size', 'Cylinders', 'Fuel_consumption_city(L/100 km)', 
                    'Fuel_consumption_hwy(L/100 km)', 'Fuel_consumption_comb(mpg)']
        # Convert numeric columns to numeric data type
        x[numeric_features] = x[numeric_features].apply(pd.to_numeric, errors='coerce')
        label_encoder = LabelEncoder()
        categorical_features = ['Make', 'Vehicle_class', 'Transmission', 'Fuel_type']
        for feature in categorical_features:
            x[feature] = label_encoder.fit_transform(x[feature])
        shap_values = explainer.shap_values(x)
        shap_explanation = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=x.iloc[0],  feature_names=feature_name)
        # Create a new figure with a larger size
        plt.figure(figsize=(10, 6))
        
        shap.plots.waterfall(shap_explanation, max_display=len(feature_name), show=False)
        plt.yticks(fontsize=12)  # Increase the font size of the y-axis labels
        plt.xticks(fontsize=12)  # Increase the font size of the x-axis labels
        plt.tight_layout()  # Adjust layout to prevent clipping
        
        # Save the plot
        plot_path = "/static/shap_plot.png"
        plt.savefig("." + plot_path)
        #Generate user-friendly explanations
        explanations = []
        for feature, shap_value in zip(feature_name, shap_values[0]):
           if shap_value > 0:
               explanation = f"The {feature} value contributed to an increase in CO2 emissions. Consider reducing {feature} to lower your carbon footprint."
               explanations.append(explanation)
        
        return render_template("resultnew.html", prediction=prediction[0], explanations=explanations, plot_path=plot_path)
if __name__ == "__main__":
    app.run(debug=False)