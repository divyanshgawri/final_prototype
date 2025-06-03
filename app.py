from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the models and encoders
model = joblib.load('/home/divyansh/Desktop/ee/prototpye_2_final/label_encoded_model.pkl')
input_encoders = joblib.load('/home/divyansh/Desktop/ee/prototpye_2_final/input_label_encoders.pkl')
output_encoders = joblib.load('/home/divyansh/Desktop/ee/prototpye_2_final/output_label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    condition = request.form['Condition']
    subtype = request.form['Subtype']

    input_data = {
        'Condition': condition,
        'Subtype': subtype
    }

    df = pd.DataFrame([input_data])

    # Encode inputs using input_encoders
    for col in ['Condition', 'Subtype']:
        df[col] = input_encoders[col].transform(df[col])

    # Predict output labels
    preds = model.predict(df)[0]

    # List of output labels in order predicted by model
    output_labels = ['Nutrient_1', 'Nutrient_2', 'Nutrient_3',
                     'Ingredient_1', 'Ingredient_2', 'Ingredient_3',
                     'Product_1', 'Product_2', 'Product_3',
                     'Type_1', 'Type_2', 'Type_3']

    # Decode each output using output_encoders
    decoded_preds = {
        label: output_encoders[label].inverse_transform([preds[i]])[0]
        for i, label in enumerate(output_labels)
    }

    return render_template('index.html', prediction=decoded_preds, inputs=input_data)

if __name__ == '__main__':
    app.run(debug=True)
#host = 0.0.0.0