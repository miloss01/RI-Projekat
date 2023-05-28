import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
from matplotlib import pyplot as plt

EPOCHS = 8
MODEL_FILE_NAME = "model"
DATA_FILE_NAME = "./train.csv"

def csv_to_model_data(csv_data):
  # csv data - matrica gde jedan red predstavlja labelu i sliku, prvi broj u redu je labela, a ostali su pikseli
  # labels - niz labela, ima ih 784
  # vectors - prvo se napravi da bude kao csv_data samo se izbaci prvi element (labela) u svakom redu,
  #           onda se svaki element podeli sa 255 da se skalira,
  #           posle se od jednodimenzionog niza napravi matrica, tako da je sada svaki red u stvari matrica 28x28,
  #           onda se sve reshape-uje da bi se prilagodilo modelu
  labels = np.array([row[0] for row in csv_data])
  vectors = np.array([(np.array(row[1:]) / 255).reshape(-1, 28) for row in csv_data]).reshape(-1, 28, 28, 1)
  return vectors, labels

def get_accuracy(labels, predictions):
  #model.evaluate(test_vectors, test_labels)
  return len([1 for i in range(len(labels)) if np.argmax(predictions[i]) == labels[i]]) / len(labels)

def get_model(vectors, labels, new):
  if not new:
    return tf.keras.models.load_model(MODEL_FILE_NAME)
  
  model = models.Sequential([
    # layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu", input_shape=(28,28,1)),
    # layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"), # izvlaci bitne feature iz slike
    layers.MaxPooling2D((2, 2)), # apstrahuje rezultate
    
    layers.Flatten(),
    # layers.Dense(32, activation="relu"), # radi dot product (input sa tezinama), dodaje bias i radi activation
    layers.Dense(10, activation="softmax")
  ])

  # model = models.Sequential([
  #   layers.Conv2D(filters=32, kernel_size=5, activation="relu"),
  #   layers.MaxPooling2D(),
  #   layers.Conv2D(filters=64, kernel_size=5, activation="relu"),
  #   layers.MaxPooling2D(),
  #   layers.Flatten(),
  #   layers.Dense(128, activation="softmax"),
  #   layers.Dense(10, activation="softmax")
  # ])

  model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
  # model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
  model.fit(vectors, labels, epochs=EPOCHS)

  model.save(MODEL_FILE_NAME)

  return model

def visual_examples(vectors, labels, predictions):
  rows = 4
  columns = 4
  fig = plt.figure(figsize=(6, 6))
  for i in range(len(vectors)):
    img = vectors[i].reshape((28, 28)) * 255
    subplot = fig.add_subplot(rows, columns, i + 1)
    subplot.set_title("L" + str(labels[i]) + "/P" + str(np.argmax(predictions[i])))
    plt.imshow(img, cmap="gray")
  plt.show()

if __name__ == "__main__":
    
    print("Loading data from csv...")
    csv = pd.read_csv(DATA_FILE_NAME)

    print("Creating training and testing data for model...")
    data = np.array(csv)

    train_csv = data[:40000]
    test_csv = data[40000:]

    train_vectors, train_labels = csv_to_model_data(train_csv)
    test_vectors, test_labels = csv_to_model_data(test_csv)

    print("Creating model...")
    model = get_model(train_vectors, train_labels, True)

    print("Making predictions for testing data...")
    predictions = model.predict(test_vectors)

    print("Getting accuracy...")
    accuracy = get_accuracy(test_labels, predictions)
    print("Accuracy: " + str(accuracy * 100) + "%")

    visual_examples(test_vectors[:10], test_labels[:10], predictions[:10])

  # data = np.array([
  #   np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 3),
  #   np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 3),
  #   np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 3)
  # ])
  # # for i in range(len(data)):
  # #   data[i].reshape(-1, 3)
  # # data.reshape(-1, 3, 3, 1)
  # data = data.reshape(-1, 3, 3, 1)
  # print(data[0][0])

  