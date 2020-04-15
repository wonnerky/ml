from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Flatten, \
    Reshape, LeakyReLU as LR, \
    Activation, Dropout
from tensorflow.keras.models import Model, Sequential
from matplotlib import pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
# Plot image data from x_train
plt.imshow(x_train[0], cmap="gray")
plt.show()
LATENT_SIZE = 32
encoder = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512),
    LR(),
    Dropout(0.5),
    Dense(256),
    LR(),
    Dropout(0.5),
    Dense(128),
    LR(),
    Dropout(0.5),
    Dense(64),
    LR(),
    Dropout(0.5),
    Dense(LATENT_SIZE),
    LR()
])
decoder = Sequential([
    Dense(64, input_shape=(LATENT_SIZE,)),
    LR(),
    Dropout(0.5),
    Dense(128),
    LR(),
    Dropout(0.5),
    Dense(256),
    LR(),
    Dropout(0.5),
    Dense(512),
    LR(),
    Dropout(0.5),
    Dense(784),
    Activation("sigmoid"),
    Reshape((28, 28))
])
img = Input(shape=(28, 28))
latent_vector = encoder(img)
output = decoder(latent_vector)
model = Model(inputs=img, outputs=output)
model.compile("nadam", loss="binary_crossentropy")
model.fit(x_train,x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test,x_test),
                )
