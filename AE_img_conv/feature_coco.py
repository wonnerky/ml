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

print(autoencoder.summary())
# check point
cp_path = 'cp/coco_cp.ckpt'
cp_dir = os.path.dirname(cp_path)
os.makedirs(cp_dir, exist_ok=True)

if os.path.isfile(cp_path):
    autoencoder.load_weights(cp_path)

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


layer_ = [autoencoder.layers[1],autoencoder.layers[3],autoencoder.layers[5]]
# layer_ = [autoencoder.layers[1],autoencoder.layers[3],autoencoder.layers[5],autoencoder.layers[7],
#                  autoencoder.layers[9],autoencoder.layers[11]]
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
    plt.imshow(display_grid, aspect='equal', cmap='viridis')
    # plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()

# plot the output from each block
# for fmap in feature_maps:
	# plot all 64 maps in an 8x8 squares
# print(K.int_shape(feature_maps))
# ix = 1
# for _ in range(2):
#     for _ in range(4):
#         # specify subplot and turn of axis
#         ax = plt.subplot(2, 4, ix)
#         plt.imshow(feature_maps[ix-1, :, :, 0])
#         # plot filter channel in grayscale
#         # plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
#         ix += 1
# # show the figure
# plt.show()