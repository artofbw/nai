# -*- coding: utf-8 -*-
'''
It is program that uses neural network to clothes classification
Authors:
Maciej Rybacki
Łukasz Ćwikliński
pip3 install -r requirements.txt
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create image plot for diagram
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


#Create plot for value in diagram
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Import dataset of fashion images with labels
fashion_mnist = tf.keras.datasets.fashion_mnist

# Divide data on train and test, divide images and labels
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Declare class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalization of data
train_images = train_images / 255.0

test_images = test_images / 255.0

# Declare model of neural network with layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Declare model compilation and train model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=10)

# Compute and print Loss and Accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# Model prediction
predictions = probability_model.predict(test_images)

# Testing third image from test dataset
i = 2
img = test_images[i]
img = (np.expand_dims(img, 0))
predictions_single = probability_model.predict(img)

# Plot and show diagram
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

# Print prediction of tested value
print(np.argmax(predictions_single[0]))
