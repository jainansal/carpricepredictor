from flask import Flask, render_template, request, redirect
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

model = pickle.load(open('lrm.pkl','rb'))
car = pd.read_csv('clean_car.csv')

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    return render_template('index.html', companies=companies,car_models=car_models,years=year,fuel_type=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=['name','company','year','kms_driven', 'fuel_type']))

    print(prediction)

    print(company,car_model,year,fuel_type,kms_driven)
    return ""

if __name__ == "__main__":
    app.run(debug=True)