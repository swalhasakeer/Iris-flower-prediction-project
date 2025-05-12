from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the flask app
app = Flask(__name__)
@app.route ('/')
def home():
    return render_template('form.html', prediction_text='')

@app.route ('/Prediction', methods=['POST'])
def predict():
    sl = float(request.form['Sepal_Length'])
    sw = float(request.form['Sepal_Width'])
    pl = float(request.form['Petal_Length'])
    pw = float(request.form['Petal_Width'])
    input_data = np.array([[sl,sw,pl,pw]])
    model = pickle.load(open("Iris_model.pkl","rb"))
    result = model.predict(input_data)
    
    if result == 0:
      prediction_text = "The Predicted Species is : Setosa"
    elif result == 1:
      prediction_text = "The Predicted Species is : Versicolor"
    elif result == 2:
      prediction_text = "The Predicted Species is : Virginica"
    else:
      prediction_text = "This species is unknown"
    
    return render_template('index.html', prediction_text = prediction_text)
if __name__ == '__main__':
    app.run(debug=True)