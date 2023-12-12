from keras import layers, models, losses


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
    out = layers.Dense(1, activation='sigmoid', name='q_values')(y)
    out = layers.Reshape((121,))(out)
    model = models.Model(inputs=input, outputs=out)
    model.summary()
    return model


model = create_model()
# Compile the model with loss
model.compile(optimizer='adam',
              loss=losses.MeanSquaredError(name="mean_squared_error"))
model.save('hex_agent_model.keras')
