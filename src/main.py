import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from clear import augment_table, tokenize
import numpy as np

train_table = pd.read_csv('data/train.csv')
test_table = pd.read_csv('data/test.csv')
label_test = pd.DataFrame(test_table.PassengerId)

m = train_table.shape[0]
train_size = (m*4)/5
# Data cleaning
train_table.loc[np.isnan(train_table.Age), 'Age'] = np.mean(train_table.Age)
test_table.loc[np.isnan(test_table.Age), 'Age'] = np.mean(train_table.Age)
test_table.loc[np.isnan(test_table.Fare), 'Fare'] = np.mean(train_table.Fare)

train_table = augment_table(train_table)
test_table = augment_table(test_table)

train_table.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_table.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

train_table, test_table = tokenize(train_table, test_table,
                                   ['Sex',
                                   'SibSp',
                                   'Parch',
                                   'Embarked',
                                   'Title',
                                   'Deck'])

Y = train_table.Survived.as_matrix()
X = train_table.drop('Survived', axis=1).as_matrix()

X_test = test_table.as_matrix()
index = np.arange(m)
np.random.shuffle(index)

X_train = X[index[:train_size]]
Y_train = Y[index[:train_size]]

X_val = X[index[train_size:]]
Y_val = Y[index[train_size:]]

# Build model
rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
# rfc = RandomForestClassifier()
parameters = {'max_features': ['auto', 1, 0.1, 0.3, 1.0],
              'max_depth': range(5, 21, 5),
              'min_samples_split': range(2, 33, 10),
              'min_samples_leaf': range(1, 32, 10)}
clf = GridSearchCV(rfc, parameters, verbose=1, n_jobs=-1)
clf.fit(X_train, Y_train)
# print clf.best_params_
# Classify
Y_hat1 = clf.predict(X_val)

prec = np.sum(Y_val==Y_hat1, dtype=np.float64)/(m-train_size)
print "Validation Precision rate: {}".format(prec)

Y_hat2 = clf.predict(X_train)

prec = np.sum(Y_train==Y_hat2, dtype=np.float64)/(train_size)
print "Train Precision rate: {}".format(prec)

Y_hat = clf.predict(X_test)
label_test = label_test.assign(Survived=Y_hat)
label_test.to_csv('output.csv', index=False)
