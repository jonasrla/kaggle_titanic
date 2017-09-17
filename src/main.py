import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from clear import augment_table, tokenize
import numpy as np

table = pd.read_csv('data/train.csv')

m = table.shape[0]
train_size = (m*4)/5
# Data cleaning
table.loc[np.isnan(table.Age), 'Age'] = np.mean(table.Age)
table = augment_table(table)
table.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
table = tokenize(table, ['Pclass',
                         'Sex',
                         'SibSp',
                         'Parch',
                         'Embarked',
                         'Title',
                         'Deck'])

Y = table.Survived.as_matrix()
X = table.drop('Survived', axis=1).as_matrix()

index = np.arange(m)
np.random.shuffle(index)

X_train = X[index[:train_size]]
Y_train = Y[index[:train_size]]

X_test = X[index[train_size:]]
Y_test = Y[index[train_size:]]
# Build model
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)

# Classify
Y_hat = rfc.predict(X_test)

prec = np.sum(Y_test==Y_hat, dtype=np.float64)/(m-train_size)
print "Precision rate: {}".format(prec)
