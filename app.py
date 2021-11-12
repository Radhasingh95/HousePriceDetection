import pickle
# save the model to disk

pickle_out = open("index_flask.pkl","wb")
pickle.dump (model, pickle_out)
loaded_model = pickle.load(open("index_flask.pkl","rb"))
result = loaded_model.score(x_test, y_test)
print(result)








#import the library files
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# initialise the Flask
app = Flask(__name__)

# define html file to get user input
@app.route('/')
def home():
    return render_template('index.html')

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,14)
    loaded_model = pickle.load(open("index_flask.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

# output page and logic
@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))

        result = ValuePredictor(to_predict_list)
        
        return render_template("result.html",prediction = result)