import os
import datetime
import numpy as np
from keras.optimizers import Adam
from keras.applications import VGG19
from keras.models import Model
from keras.layers import UpSampling2D, Conv2D, LeakyReLU
from keras.layers import BatchNormalization, Activation, Add
from keras.layers import Input, Dense
import matplotlib.pyplot as plt

# Cluster Paths
DATA_PATH = "/scratch/users/ju12/USR-248-2/train_val/"  # Replace with your dataset path
MODEL_SAVE_PATH = "/scratch/users/ju12/USR-248-2/model_checkpoints/"  # Path to save model checkpoints

# Ensure checkpoint directory exists
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# Data loader
class dataLoaderUSR:
    def __init__(self, DATA_PATH, SCALE=2):
        self.DATA_PATH = DATA_PATH
        self.SCALE = SCALE
        self.train_lr_path = os.path.join(DATA_PATH, f"lr_{SCALE}x/")
        self.train_hr_path = os.path.join(DATA_PATH, "hr/")
        self.train_images_lr = sorted(os.listdir(self.train_lr_path))
        self.train_images_hr = sorted(os.listdir(self.train_hr_path))
        self.num_train = len(self.train_images_lr)

    def load_batch(self, batch_size):
        for i in range(0, self.num_train, batch_size):
            lr_batch = []
            hr_batch = []
            for j in range(batch_size):
                if i+j >= self.num_train:
                    break
                lr_img = plt.imread(os.path.join(self.train_lr_path, self.train_images_lr[i+j]))
                hr_img = plt.imread(os.path.join(self.train_hr_path, self.train_images_hr[i+j]))
                lr_batch.append(lr_img)
                hr_batch.append(hr_img)
            yield np.array(lr_batch), np.array(hr_batch)

# Model definition (SRGAN)
class SRGAN_model:
    def __init__(self, lr_shape, hr_shape, SCALE=2):
        self.SCALE = SCALE
        self.lr_shape, self.hr_shape = lr_shape, hr_shape
        self.lr_width, self.lr_height, self.channels = lr_shape
        self.hr_width, self.hr_height, _ = hr_shape
        self.n_residual_blocks = 16
        optimizer = Adam(0.0002, 0.5)
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.disc_patch = (30, 40, 1)
        self.gf = 64
        self.df = 64
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)
        fake_hr = self.generator(img_lr)
        fake_features = self.vgg(fake_hr)
        self.discriminator.trainable = False
        validity = self.discriminator(fake_hr)
        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=optimizer)

    def build_vgg(self):
        vgg = VGG19(weights="imagenet", include_top=False)
        # Extract features from block5_conv4
        selected_layer = vgg.get_layer("block5_conv4").output
        model = Model(inputs=vgg.input, outputs=selected_layer)
        return model

    def build_generator(self):
        def residual_block(layer_input, filters):
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d
        def deconv2d(layer_input):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u
        img_lr = Input(shape=self.lr_shape)
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])
        u1 = deconv2d(c2)
        u2 = u1 if self.SCALE < 4 else deconv2d(u1)
        u3 = u2 if self.SCALE < 8 else deconv2d(u2)
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u3)
        return Model(img_lr, gen_hr)

    def build_discriminator(self):
        def d_block(layer_input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(negative_slope=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        d0 = Input(shape=self.hr_shape)
        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)
        validity = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(d8)
        return Model(d0, validity)

# Training Parameters
lr_shape = (240, 320, 3)
hr_shape = (480, 640, 3)
batch_size = 1
num_epochs = 10
data_loader = dataLoaderUSR(DATA_PATH, SCALE=2)
gan = SRGAN_model(lr_shape, hr_shape, SCALE=2)

# Training Loop
step = 0
for epoch in range(num_epochs):
    for imgs_lr, imgs_hr in data_loader.load_batch(batch_size):
        # Normalize images between -1 and 1
        imgs_lr = imgs_lr / 127.5 - 1
        imgs_hr = imgs_hr / 127.5 - 1

        # Generate high-resolution images from low-resolution images
        fake_hr = gan.generator.predict(imgs_lr)

        # Train the discriminator
        valid = np.ones((batch_size, 30, 40, 1))
        fake = np.zeros((batch_size, 30, 40, 1))
        d_loss_real = gan.discriminator.train_on_batch(imgs_hr, valid)
        d_loss_fake = gan.discriminator.train_on_batch(fake_hr, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        image_features = gan.vgg.predict(imgs_hr)
        g_loss = gan.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

        step += 1
        if step % 10 == 0:
            print(f"Epoch: {epoch+1}, Step: {step}, D Loss: {d_loss[0]}, G Loss: {g_loss[0]}")

    # Save model checkpoints every epoch
    ckpt_name = f"generator_epoch_{epoch+1}"
    gan.generator.save(os.path.join(MODEL_SAVE_PATH, f"{ckpt_name}.h5"))
    # Save model architecture as JSON
    model_json = gan.generator.to_json()
    with open(os.path.join(MODEL_SAVE_PATH, f"{ckpt_name}.json"), "w") as json_file:
        json_file.write(model_json)
    print(f"Saved generator checkpoint and architecture for epoch {epoch+1}")
