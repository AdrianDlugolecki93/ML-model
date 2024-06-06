# verify_model.py
import pickle

try:
    with open('model/perceptron_model.pkl', 'rb') as f:
        model = pickle.load(f)
        print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")