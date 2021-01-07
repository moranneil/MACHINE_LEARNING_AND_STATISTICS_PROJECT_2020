import sys 
#print("The system paths is: ",sys.path)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.linear_model as lin
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from mpl_toolkits.mplot3d import Axes3D
import os

# Load dataset from csv_file.
wind_df = pd.read_csv('./powerproduction.csv')


# https://web.microsoftstream.com/video/08404c4e-fe8b-4b84-832e-6bee57b5b160
# https://github.com/ianmcloughlin/jupyter-teaching-notebooks/blob/master/models.ipynb

def doLinearRegressionAlgorithm(wind_df, speed_input):
    
    speed_input = speed_input

    def f(speed, p):
        return p[0] + speed * p[1]

    speed = wind_df["speed"].to_numpy()
    y = wind_df["power"].to_numpy()

    def predict(speed):
        return round(f(speed, p),2)

    # print("speed is before reshape ",speed)
    speed = speed.reshape(-1, 1)
    # print("speed is after reshape ",speed)

    model = lin.LinearRegression()
    model.fit(speed, y)

    r = model.score(speed, y)
    p = [model.intercept_, model.coef_[0]]

    return(predict(float(speed_input)))





def receive_speed_from_webpage(speed_input):
    speed_input = speed_input
    return doLinearRegressionAlgorithm(wind_df,speed_input)


def model_power_MLP(wind_df, speed_input, max_iter= 10000):

    x = wind_df.iloc[:, :-1].values
    y = wind_df.iloc[:, 1].values
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    
    num_samples = wind_df.shape[0]
    cutoff = (num_samples * 3) // 4
    
    speed = wind_df["speed"].to_numpy()
    y = wind_df["power"].to_numpy()

    
    Xtrn = wind_df.drop('power', 1).iloc[:cutoff,:]
    Ytrn = wind_df['power'].iloc[:cutoff]
    Xval = wind_df.drop('power', 1).iloc[cutoff:,:]

    Yval = wind_df['power'].iloc[cutoff:]
    # model = MLPRegressor(validation_fraction = 0, solver='lbfgs', max_iter= max_iter).fit(Xtrn, Ytrn)
    model = MLPRegressor(validation_fraction = 0, solver='lbfgs', max_iter= max_iter).fit(x_train, y_train)
    print("The value of model is: ",model)
    print("x_test is: ",x_test[0])

    speed_input_array = [[speed_input]]
    print("speed_input_array is: ",speed_input_array )


    x_test2D = [x_test[0]]
    print("x_test2D is: ",x_test2D)
    print("Neural Prediction is: ",model.predict(x_test2D))
    print("Neural Prediction2 is: ",model.predict(speed_input_array))
    result = model.predict(speed_input_array)
    value = result.item(0)
    return (round(value,2))
    


def receive_speed_from_webpage_Neural(speed_input):
    speed_input = speed_input
    return model_power_MLP(wind_df, speed_input, max_iter= 10000)


