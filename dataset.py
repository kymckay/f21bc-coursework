import pandas as pd

# The input data (each row is an instance)
data = pd.read_csv("data_banknote_authentication.txt",
    sep=",",
    header=None,
)
x = data.iloc[:, 0:3].to_numpy()
y = data.iloc[:, 4].to_numpy()

# Test data set to check for overfitting (20% of overall set)
# Assuming even distribution
split_index = round(y.shape[0] * 0.2)

x_test = x[:split_index]
y_test = y[:split_index]

x = x[split_index:]
y = y[split_index:]