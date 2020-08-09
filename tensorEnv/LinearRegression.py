import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
from sklearn import linear_model
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")

# Set attributes you want to look at
data = data[["G1","G2","G3","studytime","failures","absences"]]

# attribute we want to predict
predict = "G3"

x = np.array(data.drop([predict],1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

### Run 30 times and see the best score for each model (the best fit percentage)
### Can comment out if you want to load in model and dont want to train model every single time
# best = 0
# for _ in range(30):
#     # Take attributes we want to predict and split into 4 variables
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
#     linear = linear_model.LinearRegression()
#
#     # find best fit line
#     linear.fit(x_train, y_train)
#     # accuracy of best fit line (Will change a little bit because we train it each time we run the program)
#     acc = linear.score(x_test, y_test)
#     print(acc)
#
#     # Save if score is bettter than best
#     if acc > best:
#         best = acc
#         # Save pickle file of model in directory for use
#         with open("studentmodel.pickle", "wb") as f:
#             pickle.dump(linear, f)


    
pickle_in = open("studentmodel.pickle","rb")
# load model into linear
linear = pickle.load(pickle_in)

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# Show correlation between two points (Just change p to see how it affects the final grade G3)
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()