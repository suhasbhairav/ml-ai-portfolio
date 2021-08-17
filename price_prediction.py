import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("./datasets/carprice.csv")

data = data[["symboling", "wheelbase", "carlength",
             "carwidth", "carheight", "curbweight",
             "enginesize", "boreratio", "stroke",
             "compressionratio", "horsepower", "peakrpm",
             "citympg", "highwaympg", "price"]]

X = np.array(data.drop(["price"], 1))
y = np.array(data["price"])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.score(X_test, predictions))