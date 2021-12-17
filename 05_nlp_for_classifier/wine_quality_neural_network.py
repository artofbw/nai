import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, plot_roc_curve

'''
It is program that uses neural network for classification
Authors:
Maciej Rybacki
Łukasz Ćwikliński

pip3 install -r requirements.txt


Input file columns
1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol
12. quality (score between 0 and 10)


#### Fixed acidity
    most acids involved with wine or fixed or nonvolatile (do not evaporate readily)

#### volatile acidity
    the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste
    
#### citric acid
    found in small quantities, citric acid can add 'freshness' and flavor to wines
    
#### residual sugar
    the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet

#### chlorides    
    the amount of salt in the wine
    
#### free sulfur dioxide
    the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine
    
#### total sulfur dioxide
    amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine
    
#### density
    the density of water is close to that of water depending on the percent alcohol and sugar content
    
#### pH
    describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale

#### sulphates
    a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant
'''

# Reading values from csv file
wine_data = pd.read_csv('winequality-white.csv', delimiter=";", encoding="utf-8")

#wine_data.head()

# Grouping values by quality column
wine_data.groupby('quality').count().reset_index()

# Getting only inputs
X = wine_data[
    [
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol'
    ]
]

#Getting only classification
y = wine_data['quality']

# Compute correlation of columns
corr = wine_data.corr(method="pearson")

# Creating figure object
f, ax = plt.subplots(figsize=(10, 10))

# Creating plot rectangular data as a color-encoded matrix
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot=True)

# Dividing values on train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardizing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating neural network model layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=35, input_shape=(X_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(units=52, activation='relu'),
    tf.keras.layers.Dense(units=72, activation='relu'),
    tf.keras.layers.Dense(units=52, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=52, activation='relu'),
    tf.keras.layers.Dense(units=42, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=52, activation='relu'),
    tf.keras.layers.Dense(units=52, activation='relu'),
    tf.keras.layers.Dense(units=52, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Compile model
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
cl = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60)

# Creating figure object
fig, ax = plt.subplots(figsize=(17, 8))

plt.plot(cl.history['accuracy'], label='accuracy')
plt.plot(cl.history['val_accuracy'], label='val_accuracy', linestyle='--')
plt.plot(cl.history['loss'], label='loss')
plt.plot(cl.history['val_loss'], label='val_loss', linestyle='--')
plt.legend()

# Calculate and print Loss and Accuracy
ModelLoss, ModelAccuracy = model.evaluate(X_test, y_test)

print(f'Test Loss is {ModelLoss}')
print(f'Test Accuracy is {ModelAccuracy}')

# Calculate prediction
y_pred = model.predict(X_test)
y_test_list = list(y_test)
total = len(y_test_list)
correct = 0

# Calculate and print correct to total classifications ratio
for i in range(total):
    if np.argmax(y_pred[i]) == y_test_list[i]:
        correct += 1

print(f'{correct}/{total}')
print(correct/total)

predict = model.predict(X_test)
# Create classification report
y_pred = []
for i in range(len(predict)):
    y_pred.append(np.argmax(predict[i]))

cr = classification_report(y_test, y_pred, zero_division=1)
print(cr)

p_test = model.predict(X_test).argmax(axis=1)
cm = tf.math.confusion_matrix(y_test, p_test)

# Create and show diagrams
f, ax = plt.subplots(figsize=(15, 6))
sns.heatmap(cm, annot=True, cmap='Blues', square=True, linewidths=0.01, linecolor='grey')
plt.title('Confustion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()
