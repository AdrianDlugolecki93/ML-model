# model.py
import pickle
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train perceptron model
model = Perceptron()
model.fit(X_train, y_train)

# Save the model
with open('model/perceptron_model.pkl', 'wb') as f:
    pickle.dump(model, f)
