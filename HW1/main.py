import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample,shuffle

data = pd.read_csv('train.csv')

#preprocessing data
position = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
NoYes = ['No','Yes']


data['Attribute1'] = pd.to_datetime(data['Attribute1'])
data['Attribute1'] = pd.DatetimeIndex(data['Attribute1']).month

data['Attribute8'] = data['Attribute8'].replace(position,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
data['Attribute9'] = data['Attribute9'].replace(position,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

data['Attribute20'] = data['Attribute20'].replace(NoYes,[0,1])
data['Attribute21'] = data['Attribute21'].replace(NoYes,[0,1])

for i in range(2,20):
    median = data['Attribute'+str(i)].median()
    for j in data['Attribute'+str(i)]:
        if j == np.nan:
            j = median

data = data.dropna()

majority = data[data['Attribute21'] == 0]
minority = data[data['Attribute21'] == 1]

majority_data_sampled = resample(majority, replace=False, n_samples=3000, random_state=0)
data = pd.concat([majority_data_sampled,minority])
data = shuffle(data)

X = data.drop(['Attribute21'],axis=1)
y = data['Attribute21']


X.train, X.test, y.train, y.test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

model = RandomForest(criterion='entropy', n_estimators=150, max_depth=10, n_jobs=2)
model.fit(X.train, y.train)

print(model.score(X.test, y.test))

predicts = []

test_data = pd.read_csv('test.csv')
test_data['Attribute1'] = pd.to_datetime(test_data['Attribute1'])
test_data['Attribute1'] = pd.DatetimeIndex(test_data['Attribute1']).month
test_data['Attribute8'] = test_data['Attribute8'].replace(position,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
test_data['Attribute9'] = test_data['Attribute9'].replace(position,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
test_data['Attribute20'] = test_data['Attribute20'].replace(NoYes,[0,1])
test_data['Attribute21'] = test_data['Attribute21'].replace(NoYes,[0,1])

for i in range(2,20):
    median = test_data['Attribute'+str(i)].median()
    for j in test_data['Attribute'+str(i)]:
        if j == np.nan:
            j = median

test_data = test_data.drop(['Attribute21'],axis=1)

predicts = model.predict(test_data)

#save predict to csv file with sequence
df = pd.DataFrame(predicts)
ids = range(0, len(predicts))
df_subm = pd.DataFrame({'id': ids, 'ans': predicts})
df_subm['ans'] = df_subm['ans'].replace(['No','Yes'],[0,1])
df_subm['id'] = df_subm['id'].astype(float)
print(df_subm.info())
df_subm.to_csv('submission.csv', index=False)









