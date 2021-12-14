# -*- coding: utf-8 -*-
'''
It is program that uses neural network to classification
Authors:
Maciej Rybacki
Łukasz Ćwikliński
pip3 install -r requirements.txt
'''

import tensorflow as tf

cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(24, 3, activation='relu', padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
