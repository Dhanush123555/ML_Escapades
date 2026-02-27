import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Load dataset
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

# Build the model using Sequential API
model = Sequential()
model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu', strides = (1,1), kernel_initializer = 'he_normal', bias_initializer = 'zeros', input_shape = (28,28,1)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu', strides = (1,1), kernel_initializer = 'he_normal', bias_initializer = 'zeros'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(512,activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(256,activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(10,activation = "softmax"))

#Show summary of model. It has ~3 million parameters
model.summary()

# Compile and train the model for 15 epochs - Achieves a validation accuracy of 92.4%
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
history = model.fit(x_train,y_train, epochs = 15, validation_data = (x_test,y_test))


# Plot the confusion matrix
y_pred = np.argmax(model.predict(x_test), axis =1)
confusion = confusion_matrix(y_test,y_pred)
print(f"confusion:\n {confusion}")
