# As usual, use a train/test split to get a reliable F1 score from two classifiers, and
# save it the scores in the provided dictionaries.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

clf1 = DecisionTreeClassifier()
clf1.fit(X, y)
print "Decision Tree F1 score: {:.2f}".format(f1_score(y, clf1.predict(X)))

clf2 = GaussianNB()
clf2.fit(X, y)
print "GaussianNB F1 score: {:.2f}".format(f1_score(y, clf2.predict(X)))


# My code:
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf1.fit(X_train, y_train)
clf1_score = f1_score(y_test, clf1.predict(X_test))
print "\nDecision Tree F1 score: {:.2f}".format(clf1_score)

clf2.fit(X_train, y_train)
clf2_score = f1_score(y_train, clf2.predict(X_train))
print "GaussianNB F1 score: {:.2f}\n".format(clf2_score)

F1_scores = {
    "Naive Bayes": clf2_score,
    "Decision Tree": clf1_score
}
