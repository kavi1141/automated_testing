import pickle
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
import json

model = pickle.load(open("models/model.pkl", "rb"))

# Generate some data for validation
X_test, y = make_regression(1000,n_features = 11)

# Test on the model
y_hat = model.predict(X_test)
reg = Lasso().fit(X_test, y_hat.ravel())
# Print out training r2
a = reg.score(X_test,y_hat.ravel())
with open("metrics.txt", 'w') as outfile:
        outfile.write("Testing score: %2.1f%%\n" % a)

