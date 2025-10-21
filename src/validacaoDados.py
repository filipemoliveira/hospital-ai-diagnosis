import pandas as pd

dataset = pd.read_csv("data/diabetes.csv")

print(dataset.shape())
print(dataset.info())