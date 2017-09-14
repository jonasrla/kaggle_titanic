import pandas as pd

TITLES = {'Don': 'Mr',
          'Major': 'Mr',
          'Capt': 'Mr',
          'Jonkheer': 'Mr',
          'Rev': 'Mr',
          'Col': 'Mr',
          'Master': 'Mr',
          'Countess': 'Mrs',
          'Mme': 'Mrs',
          'Mlle': 'Miss',
          'Ms': 'Miss'}

title_set = {'Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev', 'Dr', 'Ms', 'Mlle',
             'Col', 'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer'}

cabin_set = {'A', 'B', 'C', 'D', 'E', 'F', 'T', 'G'}

def find(string, str_set):
    if type(string) is str:
        for i in str_set:
            if i in string:
                return i
    return 'unknown'

unif_titles = lambda X: TITLES[X] if X in TITLES else X

def augment_table(table):
    table['Title'] = table['Name'].map(lambda x: unif_titles(find(x, title_set)))
    table['Deck'] = table['Cabin'].map(lambda x: find(x, cabin_set))
    table['Family_Size'] = table['SibSp'] + table['Parch']
    table['Age*Class'] = table['Age'] * table['Pclass']
    table['Fare_Per_Person'] = table['Fare'] / (table['Family_Size']+1)
    return table
