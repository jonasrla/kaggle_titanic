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

table.loc[np.isnan(table.Age)].Age = 29.6991

# Build model

# Classify
