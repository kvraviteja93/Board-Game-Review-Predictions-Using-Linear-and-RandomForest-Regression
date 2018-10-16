#Importing The Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#Importing The Dataset
games=pd.read_csv("games.csv")

#Exploring The Dataset
print(games.columns)
print(games.shape)

#Making a histogram of all ratings in the average rating column
plt.hist(games['average_rating'])
plt.show()

#Print the first row of all the games with zero score
print(games[games['average_rating']==0].iloc[0])

#Print the first rows of games with score greater than zero
print(games[games['average_rating']>0].iloc[0])

#Remove any rows without user rating
games=games[games['users_rated']>0]

#removing missing data
games=games.dropna(axis=0)

#Making a histogram
plt.hist(games['average_rating'])
plt.show()

#Correlation Matrix
corrmat=games.corr()
fig=plt.figure(figsize=(12,9))

sns.heatmap(corrmat,vmax=.8,square=True)
plt.show()

#Getting columns from our dataset
columns=games.columns.tolist()
columns
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]

#Storing The target/prediction variable
target = "average_rating"

#Splitting data into train test set
from sklearn.cross_validation import train_test_split
#Training Set
train=games.sample(frac=0.8, random_state=1)

#Test Set
test = games.loc[~games.index.isin(train.index)]
print(train.shape)
print(test.shape)


#Fitting Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lreg=LinearRegression()
lreg.fit(train[columns],train[target])

#Predictions on the test set
pred=lreg.predict(test[columns])
#Computing Errors
mean_squared_error(pred,test[target])

#Fitting Random Forest Regression Model
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)
rf.fit(train[columns],train[target])

# Make predictions.
predictions = rf.predict(test[columns])
# Compute the error.
mean_squared_error(predictions, test[target])

# We Can Conclude That Random Forest Regressor is Better At Predicting the Average Ratings 
#for This Particular Dataset