import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

# Set attributes you want to look at
data = data[["G1","G2","G3","studytime","failures","absences"]]

# attribute we want to predict
predict = "G3"

x = np.array(data.drop([predict],1))
y = np.array(data[predict])

# Take attributes we want to predict and split into 4 variables
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()

# find best fit line
linear.fit(x_train, y_train)
# accuracy of best fit line (Will change a little bit because we train it each time we run the program)
acc = linear.score(x_test, y_test)
print(acc)

print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])