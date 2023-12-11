from keras import layers, models
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def preprocess_input(input_array):
    # Convert the NumPy array to a list of integers
    flat_list = [int(num) for num in re.findall(r'\b\d+\b', str(input_array))]
    
    # Convert the list to a NumPy array with the desired shape
    array_2d = np.array(flat_list).reshape(2, 11, 11, 1)
    
    return array_2d

# Define the neural network architecture for board evaluation
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(2, 6, 6, 1)),
#     layers.BatchNormalization(),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(32, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(1, activation='tanh', name='board_eval'),
# ])

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(2, 11, 11, 1)),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),  
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),  
    layers.Dense(32, activation='relu'),   
    layers.Dense(1, activation='tanh', name='board_eval'),
])

# Compile the model with appropriate loss and optimizer for regression
model.compile(optimizer='adam', loss='mean_squared_error')

df = pd.read_csv(r"board_evaluation.csv")
X = df.iloc[:, :2].values
y = df.iloc[:, 2].values

# Split X and y to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.array([preprocess_input(x) for x in X_train])
X_test = np.array([preprocess_input(x) for x in X_test])

# Training
history = model.fit(X_train, y_train, epochs=35, validation_data=(X_test, y_test), batch_size=32)

# Visualize the training loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

model.save('board_evaluation_model.keras')