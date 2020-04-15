from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Flatten
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os

input_img = Input(shape=(28, 28, 1))  # 'channels_firtst'이미지 데이터 형식을 사용하는 경우 이를 적용

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# 이 시점에서 표현(representatoin)은 (4,4,8) 즉, 128 차원

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # 'channels_firtst'이미지 데이터 형식을 사용하는 경우 이를 적용
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # 'channels_firtst'이미지 데이터 형식을 사용하는 경우 이를 적용

# check point
cp_path = 'cp/cp.ckpt'
cp_dir = os.path.dirname(cp_path)
os.makedirs(cp_dir, exist_ok=True)

if os.path.isfile(cp_path):
    autoencoder.load_weights(cp_path)

cpCallback = ModelCheckpoint(cp_path, verbose=0)

# Train
if not os.path.isfile(cp_path):
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train,x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test,x_test),
                callbacks=[cpCallback]
                )

# decoded_imgs = autoencoder.predict(x_test)

layer_ = [autoencoder.layers[1],autoencoder.layers[3],autoencoder.layers[5]]
# layer_ = [autoencoder.layers[1],autoencoder.layers[3],autoencoder.layers[5],autoencoder.layers[7],
#                  autoencoder.layers[9],autoencoder.layers[11]]
layer_outputs = [layer.output for layer in layer_]
activation_model = Model(inputs=autoencoder.input, outputs=layer_outputs)
img = x_test[51].reshape(1,28,28,1)

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
            # channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            # channel_image /= (channel_image.std() + 1e-7)
            # channel_image *= 64
            # channel_image += 128
            # channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # display_grid[col * size: (col + 1) * size,  # Displays the grid
            # row * size: (row + 1) * size] = channel_image
            plt.title(layer_name)
            plt.imshow(channel_image, aspect='auto', cmap='gray')
    # scale = 1. / size
    # plt.figure(figsize=(scale * display_grid.shape[1],
    #                     scale * display_grid.shape[0]))
    # plt.title(layer_name)
    # plt.grid(False)
    # plt.imshow(display_grid, aspect='auto', cmap='gray')
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