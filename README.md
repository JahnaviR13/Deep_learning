# Importing all the necessary layers for the model 

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers import Flatten
from keras.regularizers import l2
from keras.layers import Dropout
from keras.layers import MaxPooling2D

# MNIST dataset which contains hand written digits
from keras.datasets import mnist

# Splitting the dataset into Training and testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
#print(x_train[0])

# Converting values of the images
x_train_n = x_train.astype('float32')
x_test_n = x_test.astype('float32')
x_train_norm = x_train_n / 255.0
x_test_norm = x_test_n / 255.0


# Implementation of the model 
model = Sequential()
model.add(Dense(784,activation='relu', input_shape=(28,28,1),kernel_regularizer=l2(0.001)))
model.add(Dense(256,activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

# Compiling the model
history=model.fit(x_train_norm, y_train, epochs=25, batch_size=32, verbose=1, validation_data=(x_test_norm,y_test))

test_loss,test_acc=model.evaluate(x_test_norm, y_test)

# Model description
model.summary()

# Computational Graph of the model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Printing the accuracy of the model
print("Tested Result:",test_acc*100)


# Plotting Training loss- Training Iterations
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model')
plt.ylabel('Training Loss')
plt.xlabel('Training iterations')
plt.legend(['Iter', 'Loss'], loc='upper left')
plt.show()


# Plotting Testing loss- Training Iterations
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('Model')
plt.ylabel('Testing Loss')
plt.xlabel('Training iterations')
plt.legend(['Iter', 'Loss'], loc='upper left')
plt.show()# Importing all the necessary layers for the model 

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers import Flatten
from keras.regularizers import l2
from keras.layers import Dropout
from keras.layers import MaxPooling2D

# MNIST dataset which contains hand written digits
from keras.datasets import mnist

# Splitting the dataset into Training and testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
#print(x_train[0])

# Converting values of the images
x_train_n = x_train.astype('float32')
x_test_n = x_test.astype('float32')
x_train_norm = x_train_n / 255.0
x_test_norm = x_test_n / 255.0


# Implementation of the model 
model = Sequential()
model.add(Dense(784,activation='relu', input_shape=(28,28,1),kernel_regularizer=l2(0.001)))
model.add(Dense(256,activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

# Compiling the model
history=model.fit(x_train_norm, y_train, epochs=25, batch_size=32, verbose=1, validation_data=(x_test_norm,y_test))

test_loss,test_acc=model.evaluate(x_test_norm, y_test)

# Model description
model.summary()

# Computational Graph of the model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Printing the accuracy of the model
print("Tested Result:",test_acc*100)


# Plotting Training loss- Training Iterations
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model')
plt.ylabel('Training Loss')
plt.xlabel('Training iterations')
plt.legend(['Iter', 'Loss'], loc='upper left')
plt.show()


# Plotting Testing loss- Training Iterations
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('Model')
plt.ylabel('Testing Loss')
plt.xlabel('Training iterations')
plt.legend(['Iter', 'Loss'], loc='upper left')
plt.show()
