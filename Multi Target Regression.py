# import the necessary libraries
import numpy as np
import pandas as pd

# load the dataset csv file
data = pd.read_csv("  ")


# To check columns name and null sum in the dataset
data.columns
data.isnull().sum()


# Converting csv file to DataFrame 
data = pd.DataFrame(data = data)
data.head()

# To get the matrix size
data.shape

#Taking the input columns from the dataset and splitting them up. x is a multi-column input variable with m columns.
x = data.iloc[:, :m]
x.head()
x.shape

#y is a column-based output variable with n columns.
y = data.iloc[:, m:m+n] 
y.head()
y.shape

#This library splits the dataset into 80% training data and 20% testing data
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test= train_test_split(x,y,test_size = 0.2, random_state = 0)
X_train.shape, x_test.shape


# First MTR model to train datasets and check the score
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
model.score(x_test, y_test)

# Second MTR model to train datasets and check the score
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
model.score(x_test, y_test)

# Third MTR model to train datasets and check the score
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(X_train, y_train)
model.score(x_test, y_test)

# Fourth MTR model to train datasets and check the score
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(x_test, y_test)




# Prediction of datasets using model which gives max score i.e., RandomForestRegressor commonly used 
# df1 is predicted dataframe from input test data.
# df0 is output test dataframe for visualization.
y_predict = model.predict(x_test)
df1 = pd.DataFrame(y_predict)
df1.head()
df1.shape
df0 = pd.DataFrame(y_test)
df0.head()
df0.shape


# Export dataframe to csv file
df.to_csv (r'C:\Users\admin\export_dataframe.csv', index = False, header=True)


#To check accuracy and errors
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

score = r2_score(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))

               
print('r2_score: ',score)
print('mae: ', mae)
print('mse: ',mse)
print('rmse: ',rmse)
