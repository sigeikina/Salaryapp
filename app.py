# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from flask import Flask, request, render_template
import joblib

# Create application
app = Flask(__name__)

# Bind home function to URL
@app.route('/')
def home():
    return render_template('salary_app.html')

    

@app.route('/predict', methods =['POST'])
def predict():
    # load model binaries 
    user_inputs = dict(request.form)
    model = joblib.load("static/model/model.sav")
    encoder = joblib.load("static/model/encoder.sav")
    X_scaler = joblib.load("static/model/x_scaler.sav")
    y_scaler  = joblib.load("static/model/y_scaler.sav")

    # get the user input data 

    age = user_inputs["Age"]
    gender = user_inputs["Gender"]
    city = user_inputs["City"]
    position = user_inputs["Position"]
    years = user_inputs["years of experience"]
    main_tech = user_inputs["main_tech"]

    # pressure = user_inputs["pressure"]
    # humidity = user_inputs["humidity"]
    # city_name = user_inputs["city_name"]
    
    # # store city names into a df 
    # city_input_df = pd.DataFrame({
    #     "city_name": [city_name]
    # })
     # store city names into a df 
    # city_input_df = pd.DataFrame({
    #     "city_name": [city_name]
    # })


    # cat_input_df = user_inputs[["Gender", "City", "Position", "main_tech", "Seniority level"]]
    cat_input_df = pd.DataFrame({
        "Gender": [gender],
        "City": [city],
        "Position": [position],
        "main_tech": [main_tech]
    })

    # use encoder to transform the city df 
    X_transformed = encoder.transform(cat_input_df)
    print(*encoder.categories_)
    cols = np.concatenate(encoder.categories_).ravel()
    
    cat_input_df = pd.DataFrame(columns=[cols], data=X_transformed.toarray())
    
    # store pressure and humidty into df 
    input_df = pd.DataFrame({
        "Age": [age],
        "years of experience": [years]
    })

    # combine both df's using indexes 
    df = input_df.merge(cat_input_df, left_index=True, right_index=True)

    # scale the X input df 
    X_scaled = X_scaler.transform(df)

    # obtain prediction (y) 
    prediction_scaled = model.predict(X_scaled)
    
    # scale prediction to human readable terms i.e. celcius 
    prediction = y_scaler.inverse_transform(prediction_scaled)
    return render_template('salary_app.html', result='Predicted Salary: {}'.format(np.round(np.clip(prediction[0][0], 20000,400000)),2))  



if __name__ == '__main__':
#Run the application
    app.run()
    
    