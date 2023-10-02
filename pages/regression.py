import numpy as np
import pandas as pd
from math import floor

from sklearn.linear_model import LinearRegression, ElasticNet, LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, MaxAbsScaler, RobustScaler, StandardScaler
from sklearn.feature_selection import SelectPercentile, f_regression

import streamlit as st
# from streamlit.hello.utils import show_code

filepath = "https://s3.amazonaws.com/talent-assets.datacamp.com/university_enrollment_2306.csv"
df = pd.read_csv(filepath)
values = {"course_type": 'classroom', "year": 2011, "enrollment_count": 0, "pre_score": '0', "post_score": 0, "pre_requirement": 'None', "department": 'unknown'}
df = df.fillna(value = values)
df['pre_score'] = df['pre_score'].replace('-', '0')
df['pre_score'] = df['pre_score'].astype(float)
df['department'] = df['department'].str.strip().replace('Math', 'Mathematics')
df_dummy = pd.get_dummies(df, drop_first = True)
X = df_dummy.drop(['enrollment_count', 'course_id'], axis = 1).values
y = df_dummy['enrollment_count'].values


def get_classifier(clf_name, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)
    if clf_name == 'Linear Regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test) # Baseline
        y_pred_train = model.predict(X_train) # Baseline
        rmse_train = MSE(y_train, y_pred_train, squared=False)
        rmse_test = MSE(y_test, y_pred, squared=False)
    elif clf_name == 'Elastic Net':
        alpha = st.sidebar.number_input('Penalty Terms Multiplier', format = '%f', value = 1.0)
        model = ElasticNet(alpha = alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train) # Baseline
        rmse_train = MSE(y_train, y_pred_train, squared=False)
        rmse_test = MSE(y_test, y_pred, squared=False)
    elif clf_name == 'TPOT-Optimized Pipeline':
        model = make_pipeline(
            SelectPercentile(score_func=f_regression, percentile=84),
            StandardScaler(),
            Binarizer(threshold=0.55),
            RobustScaler(),
            MaxAbsScaler(),
            LassoLarsCV(normalize=False)
            )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train) # Baseline
        rmse_train = MSE(y_train, y_pred_train, squared=False)
        rmse_test = MSE(y_test, y_pred, squared=False)
    return model, rmse_train, rmse_test

random_state = st.sidebar.number_input('Random Seed', value = 42, step = 1)
clf = st.sidebar.selectbox("Regression Model", ('Elastic Net', 'Linear Regression', 'TPOT-Optimized Pipeline'))
model, rmse_train, rmse_test = get_classifier(clf, random_state)
act = st.sidebar.radio("Action", ('Training Information', 'Make Prediction'))

def get_action(act_name, model):
    if act_name == 'Training Information':
        st.markdown(
            """ 
            # Trained Model Information
            """
            )
        st.write('Hyperparameters of the Selected Model', model.get_params())
        st.write('Root Mean Square Error on the Training Set:', rmse_train)
        st.write('Root Mean Square Error on the Testing Set:', rmse_test)
    else:
        st.write("### Input Features")
        t= str.lower(st.selectbox('Course Type', ('Online', 'Classroom')))
        y = st.number_input('Year', value = 2011)
        pr = st.number_input('Pre-assessment Score', format = '%f', value = 84.0)
        po = st.number_input('Post-Assessment Score', format = '%f', value = 95.0)
        pre = st.selectbox('Pre-requisite', ('None', 'Beginner', 'Intermediate'))
        d = st.selectbox('Department', ('Science', 'Mathematics', 'Technology', 'Engineering'))
        if t == 'online':
            t = 1
        else:
            t = 0
        if pre == 'Beginner':
            pre = np.zeros(2)
        elif pre == 'Intermediate':
            pre = [1,0]
        else:
            pre = [0,1]
        if d == 'Engineering':
            d = np.zeros(3) 
        elif d == 'Mathematics':
            d = [1,0,0]
        elif d == 'Science':
            d = [0,1,0]
        else:
            d = [0,0,1]
        input_feat = np.array(np.concatenate([[y,pr,po,t],pre,d]).reshape(1,-1))
        st.write('#### Predicted Enrollment Count:', floor(model.predict(input_feat)[0]))
    return

get_action(act, model)