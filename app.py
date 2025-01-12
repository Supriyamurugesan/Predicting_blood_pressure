from flask import Flask, request, render_template
import joblib
import numpy as np


systolic_model = joblib.load('bp_model_systolic.pkl')
diastolic_model = joblib.load('bp_model_diastolic.pkl')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        heart_rate = float(request.form['heart_rate'])
        
        
        bmi = weight / ((height / 100) ** 2)
        
        
        input_data = np.array([[age, bmi, heart_rate]])
        
        
        systolic_bp = systolic_model.predict(input_data)[0]
        diastolic_bp = diastolic_model.predict(input_data)[0]

       
        return render_template(
            'index.html',
            systolic_bp=f"{systolic_bp:.2f}",
            diastolic_bp=f"{diastolic_bp:.2f}",
            inputs={
                'age': age, 'height': height, 'weight': weight, 'heart_rate': heart_rate
            }
        )
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
