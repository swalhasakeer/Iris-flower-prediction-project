import pickle
import numpy
model = pickle.load(open("Iris_model.pkl","rb"))
sample_input = [[6.7,3.0,5.2,2.3]]
prediction = model.predict(sample_input)
print(prediction)