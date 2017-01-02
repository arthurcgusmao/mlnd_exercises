# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

clf1 = DecisionTreeClassifier()
clf1.fit(X, y)
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(recall(y,clf1.predict(X)),precision(y,clf1.predict(X)))

clf2 = GaussianNB()
clf2.fit(X, y)
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(recall(y,clf2.predict(X)),precision(y,clf2.predict(X)))


# My code:
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)


clf1.fit(X_train, y_train)
clf1_recall = recall(y_test, clf1.predict(X_test))
clf1_precision = precision(y_test, clf1.predict(X_test))
print "\nDecision Tree recall: {:.2f} and precision: {:.2f}".format(clf1_recall, clf1_precision)

clf2.fit(X_train, y_train)
clf2_recall = recall(y_test, clf2.predict(X_test))
clf2_precision = precision(y_test, clf2.predict(X_test))
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(clf2_recall, clf2_precision)


results = {
    "Naive Bayes Recall": clf2_recall,
    "Naive Bayes Precision": clf2_precision,
    "Decision Tree Recall": clf1_recall,
    "Decision Tree Precision": clf1_precision
}
