from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, load_img
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import shutil
import os

training_dir = 'data/val2017'
test_dir = 'data/val2017/test'
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(training_dir,
                                                 target_size=(200,200),
                                                 batch_size=32,
                                                 class_mode='input')
test_set = train_datagen.flow_from_directory(test_dir,
                                                 target_size=(200,200),
                                                 batch_size=32,
                                                 class_mode='input')
training_set.image_data_generator
input_img = Input(shape=(200, 200, 3))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
ae_filter = Model(input_img, decoded)

print(autoencoder.summary())
# check point
cp_path = 'cp/coco_cp.ckpt'
cp_dir = os.path.dirname(cp_path)
os.makedirs(cp_dir, exist_ok=True)

if os.path.isfile(cp_path):
    autoencoder.load_weights(cp_path)
    ae_filter.load_weights(cp_path)

cpCallback = ModelCheckpoint(cp_path, verbose=0)

# Train
if not os.path.isfile(cp_path):
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit_generator(training_set,
                steps_per_epoch=4500/32,
                epochs=15,
                shuffle=True,
                validation_data=test_set,
                validation_steps=500/32,
                callbacks=[cpCallback]
                )


layer_ = [autoencoder.layers[1],autoencoder.layers[3],autoencoder.layers[5],autoencoder.layers[7],
                 autoencoder.layers[9],autoencoder.layers[11]]
layer_outputs = [layer.output for layer in layer_]
activation_model = Model(inputs=autoencoder.input, outputs=layer_outputs)

img = Image.open('data/val2017/000000001584.jpg')
img = img.resize((200,200))
print(img.size)
img = np.asarray(img)
img = img / 255.
img = img.reshape(1,200,200,3)
fig = plt.figure(figsize=(5,5))
plt.imshow(img[0,:,:,:])
plt.show()
result = autoencoder.predict(img)
plt.imshow(result[0,:,:,:])

activations = activation_model.predict(img)

layer_names = []
for layer in layer_:
    layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

images_per_row = 8

for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
    n_features = layer_activation.shape[-1]  # Number of features in the feature map
    size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):  # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            channel_image /= (channel_image.std() + 1e-7)
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # Displays the grid
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='equal')     #, cmap='viridis')
plt.show()


# -------------------------------------------------
# Utility function for displaying filters as images
# -------------------------------------------------

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    # x *= 255
    # x = np.clip(x, 0, 255).astype('uint8')
    return x


# ---------------------------------------------------------------------------------------------------
# Utility function for generating patterns for given layer starting from empty input image and then
# applying Stochastic Gradient Ascent for maximizing the response of particular filter in given layer
# ---------------------------------------------------------------------------------------------------

def generate_pattern(layer_name, filter_index, size=25):
    layer_output = ae_filter.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, ae_filter.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([ae_filter.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(80):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


# ------------------------------------------------------------------------------------------
# Generating convolution layer filters for intermediate layers using above utility functions
# ------------------------------------------------------------------------------------------

layer_name = 'conv2d_4'
size = 200
margin = 5
results = np.zeros((4 * size + 7 * margin, 8 * size + 7 * margin, 3))

for i in range(4):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + j, size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

plt.figure(figsize=(20, 20))
plt.imshow(results, aspect='equal')
plt.show()