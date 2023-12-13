from keras import layers, models,losses
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def preprocess_input(input_array):
    # Convert the NumPy array to a list of integers
    flat_list = [int(num) for num in re.findall(r'\b\d+\b', str(input_array))]
    
    # Convert the list to a NumPy array with the desired shape
    array_2d = np.array(flat_list).reshape(11, 11, 1)
    
    return array_2d

def create_model():
    input_shape = (11, 11, 1)
    input = layers.Input(shape=input_shape)
    a = layers.Conv2D(49, (5, 5), activation='relu', padding='same')(input)
    b = layers.Conv2D(81, (3, 3), activation='relu', padding='same')(input)
    sub = layers.Concatenate()([a, b])
    y = layers.BatchNormalization(epsilon=1e-5)(sub)
    for i in range(1, 4):
        a = layers.Conv2D(49, (5, 5), activation='relu', padding='same')(y)
        b = layers.Conv2D(81, (3, 3), activation='relu', padding='same')(y)
        sub = layers.Concatenate()([a, b])
        y = layers.BatchNormalization(epsilon=1e-5)(sub)
    for i in range(4, 8):
        y = layers.Conv2D(130, (3, 3), activation='relu', padding='same')(y)
        y = layers.BatchNormalization(epsilon=1e-5)(y)
    y = layers.Conv2D(130, (5, 5), activation='relu')(y)
    y = layers.Conv2D(130, (5, 5), activation='relu')(y)
    y = layers.Conv2D(130, (3, 3), activation='relu')(y)
    out = layers.Dense(1, activation='tanh', name='board_score')(y)
    model = models.Model(inputs=input, outputs=out)
    model.summary()
    return model

def create_model2():
    input_shape = (11, 11, 1)  # Assuming a single channel input. Change if you have multiple channels
    num_filters = 32           # You can adjust the number of filters
    kernel_size = (3, 3)       # Kernel size for the convolutional layers
    pool_size = (2, 2)         # Pool size for the max pooling layers
    dropout_rate = 0.5         # Dropout rate for the dropout layers
    num_dense_neurons = 64     # Number of neurons in the dense layer

    # Create the model
    model = Sequential()

    # First Convolutional Block
    model.add(Conv2D(num_filters, kernel_size, input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout_rate))

    # Second Convolutional Block (optional, you can add more blocks as needed)
    model.add(Conv2D(num_filters * 2, kernel_size, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout_rate))

    # Flattening the Convolutional Layers
    model.add(Flatten())

    # Dense Layers
    model.add(Dense(num_dense_neurons, activation='relu'))
    model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(1, activation='tanh'))  # tanh activation for output between -1 and 1

    model.summary()

    return model

model = create_model()

# Compile the model with appropriate loss and optimizer for regression
model.compile(optimizer='adam',
              loss=losses.MeanSquaredError(name="mean_squared_error"))

df = pd.read_csv(r"board_evaluation.csv")

X = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# Split X and y to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.array([preprocess_input(x) for x in X_train])
X_test = np.array([preprocess_input(x) for x in X_test])

# Training
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32)

# Visualize the training loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

model.save('board_evaluation_model.keras')