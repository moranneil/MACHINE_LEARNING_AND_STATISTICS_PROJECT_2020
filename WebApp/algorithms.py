import sys 
#print("The system paths is: ",sys.path)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.neural_network as sknn
import sklearn.linear_model as lin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from mpl_toolkits.mplot3d import Axes3D
import os

# Load dataset from csv_file.
wind_df = pd.read_csv('powerproduction.csv')

# Clean wind_df Dataset by removing power values that are not equal to 0 and assign it to a new Dataframe cleaned_wind_df
cleaned_wind_df = wind_df.loc[wind_df['power'] != 0 ]

# The linear regression code was party taken from Ian McLoughlins Lecture with modifications
# https://web.microsoftstream.com/video/08404c4e-fe8b-4b84-832e-6bee57b5b160
# https://github.com/ianmcloughlin/jupyter-teaching-notebooks/blob/master/models.ipynb

def modelLinearRegression(wind_df, speed_input):
    
    speed_input = speed_input

    speed = wind_df["speed"].to_numpy()
    y = wind_df["power"].to_numpy()

    # print("speed is before reshape ",speed)
    speed = speed.reshape(-1, 1)
    # print("speed is after reshape ",speed)

    model = lin.LinearRegression()
    model.fit(speed, y)

    speed_input_array = [[speed_input]]
    print("speed_input_array is: ",speed_input_array )

    result = model.predict(speed_input_array)
    value = result.item(0)

    r = model.score(speed, y)
    p = [model.intercept_, model.coef_[0]]

    return (round(value,2))

    # return(predict(float(speed_input)))

# This is a calling function or the WebApp for Linear Regression Algorithm
# There is also an if statement that checks if the inputted Wind Speed is in the given power range 0.325 and 24.399
def receive_speed_from_webpage(speed_input):
    speed_input = speed_input
    # If speed from Webpage is between 0.325 and 24.399 then run it through the algorithm if not return 0
    if (speed_input >= 0.325 and speed_input <= 24.399):
        # print("Running Wind Speed Through Algorithm")
        return modelLinearRegression(wind_df,speed_input)
    else:
        # print("Returning 0")
        return 0

# The Neural network algorithm was taken and developed from this example below and from the code format from the Linear Regression algorithm    
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
# https://towardsdatascience.com/ml-preface-2-355b1775723e

def modelNeuralMLP(wind_df, speed_input):

    x = wind_df.iloc[:, :-1].values
    y = wind_df.iloc[:, 1].values
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    y = wind_df["power"].to_numpy()

    modelN = sknn.MLPRegressor(max_iter= 10000).fit(x_train, y_train)
    # print("The value of model is: ",model)
    # print("x_test is: ",x_test[0])

    speed_input_array = [[speed_input]]
    print("speed_input_array is: ",speed_input_array )

    result = modelN.predict(speed_input_array)
    value = result.item(0)
    return (round(value,2))

# This is a calling function or the WebApp for Neural Network Algorithm
def receive_speed_from_webpage_Neural(speed_input):
    speed_input = speed_input
    # If speed from Webpage is between 0.325 and 24.399 then run it through the algorithm if not return 0
    if (speed_input >= 0.325 and speed_input <= 24.399):
        # print("Running Wind Speed Through Algorithm")
        return modelNeuralMLP(wind_df, speed_input)
    else:
        # print("Returning 0")
        return 0

# This is the same Neural Network Algorithm but on the cleansed Dataset
# There is also an if statement that checks if the inputted Wind Speed is in the given power range 0.325 and 24.399
def modelNeuralMLPClean(cleaned_wind_df, speed_input):

    x_c = cleaned_wind_df.iloc[:, :-1].values
    y_c = cleaned_wind_df.iloc[:, 1].values
    
    x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(x_c, y_c, test_size=0.3, random_state=0)
    
    # num_samples = cleaned_wind_df.shape[0]
    # cutoff = (num_samples * 3) // 4
    
    # speed = cleaned_wind_df["speed"].to_numpy()
    # y = cleaned_wind_df["power"].to_numpy()

    modelNC = sknn.MLPRegressor(max_iter= 10000).fit(x_train_c, y_train_c)
    # print("The value of model is: ",model)
    # print("x_test_c is: ",x_test_c[0])

    speed_input_array = [[speed_input]]
    print("speed_input_array is: ",speed_input_array )

    result = modelNC.predict(speed_input_array)
    value = result.item(0)
    return (round(value,2))

# This is a calling function or the WebApp for Neural Network Algorithm 
# There is also an if statement that checks if the inputted Wind Speed is in the given power range 0.325 and 24.399
def receive_speed_from_webpage_Neural_Clean(speed_input):
    print("Before speed_input = speed_input",type(speed_input))
    speed_input = speed_input
    # If speed from Webpage is between 0.325 and 24.399 then run it through the algorithm if not return 0
    if (speed_input >= 0.325 and speed_input <= 24.399):
        # print("Running Wind Speed Through Algorithm")
        return modelNeuralMLPClean(cleaned_wind_df, speed_input)
    else:
        # print("Returning 0")
        return 0


