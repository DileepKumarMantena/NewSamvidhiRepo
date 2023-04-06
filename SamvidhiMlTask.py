import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as ml

path = "C:/Users/deeli/OneDrive/Desktop/data.csv"
df = pd.read_csv(path)


dataset = np.loadtxt(df, delimiter=",")
print("DataSet ",dataset)
X, y = dataset[0],dataset[1]
print("++++++++++",X,y)
k_folds = KFold(n_splits = 2)
clf = DecisionTreeClassifier(random_state=42)
print("Clf",clf)

xpoints = X
ypoints = y

ml.plot(xpoints, ypoints)
ml.show()
