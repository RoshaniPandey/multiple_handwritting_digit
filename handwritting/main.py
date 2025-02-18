from scipy.io import loadmat
import numpy as np
from model import neural_network
from Randlnitialise import initialise
from Prediction import predict
from scipy.optimize import minimize
import os

# Load MNIST dataset
data = loadmat('mnist-original.mat')
X = data['data'].T  # Transpose to correct shape
y = data['label'].flatten()

# Normalize dataset
X = X / 255.0

# Split into training and testing sets
X_train, X_test = X[:60000, :], X[60000:, :]
y_train, y_test = y[:60000], y[60000:]

# Neural network parameters
input_layer_size = 784
hidden_layer_size = 100
num_labels = 10

# Initialize weights with small random values
initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
initial_Theta2 = initialise(num_labels, hidden_layer_size)

# Flatten Theta values for optimization
initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
lambda_reg = 0.1
maxiter = 500  # Increased iterations

print("Training the neural network, please wait...")
results = minimize(neural_network, x0=initial_nn_params, args=(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg), 
                   options={'disp': True, 'maxiter': maxiter}, method="L-BFGS-B", jac=True)

# Extract trained weights
nn_params = results["x"]
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))

# Save trained weights
np.savetxt('Theta1.txt', Theta1, delimiter=' ')
np.savetxt('Theta2.txt', Theta2, delimiter=' ')
print("Training complete! Weights saved.")

# Test accuracy
pred_train = predict(Theta1, Theta2, X_train)
pred_test = predict(Theta1, Theta2, X_test)

print(f'Training Accuracy: {np.mean(pred_train == y_train) * 100:.2f}%')
print(f'Test Accuracy: {np.mean(pred_test == y_test) * 100:.2f}%')
