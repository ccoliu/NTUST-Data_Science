import numpy as np
import pandas as pd
import torch

data = pd.read_csv('train.csv')

features = data.drop(['Attribute21'],axis=1)
labels = data['Attribute21']

#preprocessing data
position = ['N','W','S','E','NW','NE','SW','SE','ESE','WNW','WSW','NNW','SSE','SSW','ENE','NNE']
NoYes = ['No','Yes']

features['Attribute6'] = features['Attribute6'].fillna(0.0)
features['Attribute7'] = features['Attribute7'].fillna(0.0)
features['Attribute16'] = features['Attribute16'].fillna(0.0)
features['Attribute17'] = features['Attribute17'].fillna(0.0)

features['year'] = features['Attribute1'].str.split('-').str[0]
features['month'] = features['Attribute1'].str.split('-').str[1]
features['day'] = features['Attribute1'].str.split('-').str[2]

features['Attribute8'] = features['Attribute8'].replace(position,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
features['Attribute9'] = features['Attribute9'].replace(position,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

features['Attribute20'] = features['Attribute20'].replace(NoYes,[0,1])

features = features.drop(['Attribute1'],axis=1)

labels = labels.replace(NoYes,[0,1])


features.tensor = torch.tensor(features.value,dtype=torch.float32)
labels.tensor = torch.tensor(labels.value,dtype=torch.float32)


