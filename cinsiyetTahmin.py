import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

#import data set file
data=pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')

#get  values from data
gender=data.iloc[:,:1].values
height=data.iloc[:,1:2]
weight=data.iloc[:,2:3]

#encoder:  Categoric -> Numeric
le=LabelEncoder()
gender[:,0]=le.fit_transform(gender[:,0])

#combine values
s=pd.concat([height,weight],axis=1).values
#Decision Tree Regressor
r_dt=DecisionTreeRegressor(random_state=0)
#train model
r_dt.fit(s,gender)
#predict Gender
genderPredict=r_dt.predict(s)
#predict new value
print(r_dt.predict([[183,64]]))


