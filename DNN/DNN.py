from pyexpat import model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('/Users/khaledelshamy/Desktop/DeepLearning_Projects/DNN/Airline_Delay_Cause.csv')
print(data)
data.info()

# remove some features
data = data.drop(['carrier','carrier_name','airport','airport_name'],axis=1)
print(data)
data.info()

# remove null
data.dropna(inplace=True)
data.info()

print(data['weather_delay'].min(), data['weather_delay'].max())

# Feature Engineering:- create a new column called weather_delay_case
data['weather_delay_case'] = data['weather_delay'].apply(lambda x: 1 if x > 100 else 0)
print(data['weather_delay_case'].value_counts())


# input features
x = data.drop(['weather_delay_case'],axis=1)

# output
y = data['weather_delay_case']
print(y)

## split training data and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=44,shuffle=True)

print('X_train shape is ', x_train)
print('X_test shape is ', x_test)
print('Y_train shape is ', y_train)
print('Y_test shape is ', y_test)

# building neural network
import tensorflow as tf
import keras

model = keras.models.Sequential([
    keras.layers.Input(shape=17),
    keras.layers.Dense(8,activation='tanh'),
    #keras.layers.Dropout(0.1),
    keras.layers.Dense(128,activation='tanh'),
    #keras.layers.Dropout(0.3),
    keras.layers.Dense(64,activation='tanh'),
    keras.layers.Dense(32,activation='tanh'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# training NN
history = model.fit(x_train,y_train,
                         validation_data=(x_test,y_test),
                         epochs=100,
                         batch_size=10000,
                         verbose=1,
                         callbacks=[tf.keras.callbacks.EarlyStopping(
                                            patience=10,
                                            monitor='val_accuracy',#"val_loss",
                                            restore_best_weights=True)])

print(model.summary())
# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_acc)

training_loss = history.history['loss']
training_accuracy = history.history['accuracy']

model.save('KerasModel.model')
newModel = keras.models.load_model('KerasModel.model')

y_pred = newModel.predict(x_test)

ModelLoss, ModelAccuracy = newModel.evaluate(x_test, y_test)
print('Model Loss is {}'.format(ModelLoss))
print('Model Accuracy is {}'.format(ModelAccuracy ))

y_pred = [np.round(i[0]) for i in y_pred]
print(y_pred)

# Plotting "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Performance Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True,cmap='Blues_r')
plt.show()

