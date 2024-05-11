import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

dataset=pd.read_csv("breastcancerdetectionproject.csv")
print(dataset.head())

# print("Non missing values:",str(dataset.isnull().shape[0]))
# print("Missing values:",str(dataset.shape[0]-dataset.isnull().shape[0]))

y=dataset["fractal_dimension_worst"]#TARGET
x=dataset.iloc[:,0:31]#FETAUREScas

# from sklearn.preprocessing import LabelEncoder

# Assuming 'fractal_dimension_worst' contains non-numeric values like 'M' or 'B'
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# label_encoder = LabelEncoder()
# x_encoded = label_encoder.fit_transform(x)


# Now 'y_encoded' can be used as the target variable for KNN
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)
y_res = knn.predict(x_test)

# x=dataset.data
# y=dataset.target

# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
# knn=KNeighborsClassifier(n_neighbors=4)
# knn.fit(x_train,y_train)
# y_res=knn.predict(x_test)