# In this exercise, we'll use the Titanic dataset as before, train two classifiers and
# look at their confusion matrices. Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

import numpy as np
import pandas as pd

# Load the dataset
from sklearn import datasets

X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the default settings for train_test_split (or test_size = 0.25 if specified).
# Then, train and test the classifiers with your newly split data instead of X and y.

clf1 = DecisionTreeClassifier()
clf1.fit(X,y)
print "Confusion matrix for this Decision Tree:\n",confusion_matrix(y,clf1.predict(X))

clf2 = GaussianNB()
clf2.fit(X,y)
print "GaussianNB confusion matrix:\n",confusion_matrix(y,clf2.predict(X))

#TODO: store the confusion matrices on the test sets below

# My code:
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)


clf1.fit(X_train,y_train)
print "\nConfusion matrix for this Decision Tree:\n",confusion_matrix(y_test,clf1.predict(X_test))

clf2.fit(X_train,y_train)
print "GaussianNB confusion matrix:\n",confusion_matrix(y_test,clf2.predict(X_test))

print "\nInformation about the number of testing data points:", len(y_test), "\n"

clf1_cm = confusion_matrix(y_test, clf1.predict(X_test))
clf2_cm = confusion_matrix(y_test, clf2.predict(X_test))

confusions = {
    "Naive Bayes": clf1_cm,
    "Decision Tree": clf2_cm
}
