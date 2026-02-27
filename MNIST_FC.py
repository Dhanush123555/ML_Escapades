import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model,Input
from tensorflow.keras.datasets import mnist

# Load mnist dataset
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape)

# Create and chain all layers
input = Input(shape = (28,28), name = "input_layer")
flatten_output = Flatten()(input)
dense1_output = Dense(64, activation = "relu", name = "hidden_layer")(flatten_output)
output = Dense(10,activation = "softmax",name = "output_layer")(dense1_output)

# Create and plot model
model = Model(inputs = input, outputs = output, name = "Simple_NN_Model")
tf.keras.utils.plot_model(model,show_shapes = True)

# Compile and train model
model.compile(optimizer= "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(x_train,y_train,epochs = 20,verbose = 2, batch_size = 1000, validation_split = 0.2)

# Evaluate model
model.evaluate(x_test,y_test)
