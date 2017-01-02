import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

reg1 = DecisionTreeRegressor()
reg1.fit(X, y)
print "Decision Tree mean absolute error: {:.2f}".format(mae(y,reg1.predict(X)))

reg2 = LinearRegression()
reg2.fit(X, y)
print "Linear regression mean absolute error: {:.2f}".format(mae(y,reg2.predict(X)))


# My code:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

reg1.fit(X_train, y_train)
reg2.fit(X_train, y_train)

reg1_mae = mae(y_test,reg1.predict(X_test))
reg2_mae = mae(y_test,reg2.predict(X_test))

print "\nDecision Tree mean absolute error: {:.2f}".format(reg1_mae)
print "Linear regression mean absolute error: {:.2f}".format(reg2_mae)

results = {
    "Decision Tree": reg1_mae,
    "Linear Regression": reg2_mae
}
