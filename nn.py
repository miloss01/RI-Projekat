import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

EPOCHS = 500
LEARNING_RATE = 0.1
DATA_FILE_NAME = "./train.csv"

def init_params():
  hidden_layer_weights = np.random.rand(10, 784) - 0.5
  hidden_layer_biases = np.random.rand(10, 1) - 0.5
  output_layer_weights = np.random.rand(10, 10) - 0.5
  output_layer_biases = np.random.rand(10, 1) - 0.5
  return hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases

def ReLU(matrix):
  return np.maximum(matrix, 0)

def ReLU_deriv(matrix):
  return matrix > 0

def softmax(matrix):
  return np.exp(matrix) / sum(np.exp(matrix))

def forward_prop(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases, input_data):
  hidden_layer_dot_product = hidden_layer_weights.dot(input_data) + hidden_layer_biases
  hidden_layer_output = ReLU(hidden_layer_dot_product)
  output_layer_dot_product = output_layer_weights.dot(hidden_layer_output) + output_layer_biases 
  output_layer_output = softmax(output_layer_dot_product)
  return hidden_layer_dot_product, hidden_layer_output, output_layer_dot_product, output_layer_output

def one_hot(labels):
  one_hot_Y = np.zeros((labels.size, 10))
  one_hot_Y[np.arange(labels.size), labels] = 1
  return one_hot_Y.T

def backward_prop(Z1, A1, A2, W2, X, labels, m):
  dZ2 = A2 - one_hot(labels)

  db2 = np.sum(dZ2) / m
  dW2 = dZ2.dot(A1.T) / m

  dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)

  dW1 = dZ1.dot(X.T) / m
  db1 = np.sum(dZ1) / m
  return dW1, db1, dW2, db2

def get_predictions(output_layer):
  return np.argmax(output_layer, 0)

def get_accuracy(predictions, labels):
  print(predictions, labels)
  return np.sum(predictions == labels) / labels.size

def display_training(output_layer_output, labels):
  predictions = get_predictions(output_layer_output)
  print(get_accuracy(predictions, labels))

def gradient_descent(input_data, labels, learning_rate, iterations, m):
  hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases = init_params()

  for i in range(iterations):
    hidden_layer_dot_product, hidden_layer_output, output_layer_dot_product, output_layer_output = forward_prop(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases, input_data)

    dW1, db1, dW2, db2 = backward_prop(hidden_layer_dot_product, hidden_layer_output, output_layer_output, output_layer_weights, input_data, labels, m)

    hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases = update_params(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases, dW1, db1, dW2, db2, learning_rate)

    if i % 10 == 0:
      print("Iteration: ", i)
      display_training(output_layer_output, labels)

  return hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases

def update_params(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases, dW1, db1, dW2, db2, alpha):
  hidden_layer_weights = hidden_layer_weights - alpha * dW1
  hidden_layer_biases = hidden_layer_biases - alpha * db1    
  output_layer_weights = output_layer_weights - alpha * dW2  
  output_layer_biases = output_layer_biases - alpha * db2    
  return hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases

def make_predictions(input_data, hidden_layer_weights_trained, hidden_layer_biases_trained, output_layer_weights_trained, output_layer_biases_trained):
  _, _, _, output_layer_output = forward_prop(hidden_layer_weights_trained, hidden_layer_biases_trained, output_layer_weights_trained, output_layer_biases_trained, input_data)
  predictions = get_predictions(output_layer_output)
  return predictions

def test_prediction(index, hidden_layer_weights_trained, hidden_layer_biases_trained, output_layer_weights_trained, output_layer_biases_trained, test_data, test_labels):
  current_image = test_data[:, index, None]

  prediction = make_predictions(test_data[:, index, None], hidden_layer_weights_trained, hidden_layer_biases_trained, output_layer_weights_trained, output_layer_biases_trained)

  label = test_labels[index]

  print("Prediction: ", prediction)
  print("Label: ", label)
  
  current_image = current_image.reshape((28, 28)) * 255
  plt.gray()
  plt.imshow(current_image, interpolation='nearest')
  plt.show()

if __name__ == "__main__":
  
  print("Loading data from csv...")
  data = pd.read_csv(DATA_FILE_NAME)

  print("Creating training and testing data for model...")
  data = np.array(data)
  num_rows, num_cols = data.shape
  np.random.shuffle(data)

  data_dev = data[0:1000].T
  labels_dev = data_dev[0]
  pixels_dev = data_dev[1:num_cols] / 255.

  data_train = data[1000:num_rows].T
  labels_train = data_train[0]
  pixels_train = data_train[1:num_cols] / 255.
  _, m_train = pixels_train.shape

  print("Creating model...")
  hidden_layer_weights_trained, hidden_layer_biases_trained, output_layer_weights_trained, output_layer_biases_trained = gradient_descent(pixels_train, labels_train, LEARNING_RATE, EPOCHS, num_rows)

  print("Making predictions for testing data...")
  test_prediction(0, hidden_layer_weights_trained, hidden_layer_biases_trained, output_layer_weights_trained, output_layer_biases_trained, pixels_dev, labels_dev)
  test_prediction(1, hidden_layer_weights_trained, hidden_layer_biases_trained, output_layer_weights_trained, output_layer_biases_trained, pixels_dev, labels_dev)
  test_prediction(2, hidden_layer_weights_trained, hidden_layer_biases_trained, output_layer_weights_trained, output_layer_biases_trained, pixels_dev, labels_dev)
  test_prediction(3, hidden_layer_weights_trained, hidden_layer_biases_trained, output_layer_weights_trained, output_layer_biases_trained, pixels_dev, labels_dev)
  test_prediction(4, hidden_layer_weights_trained, hidden_layer_biases_trained, output_layer_weights_trained, output_layer_biases_trained, pixels_dev, labels_dev)