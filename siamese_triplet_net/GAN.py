import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# تعریف پارامترهای شبکه
latent_dim = 128
height = 128
width = 128
channels = 3

# ساختار ژنراتور
generator_input = tf.keras.Input(shape=(latent_dim,))

# اولین لایه که یک لایه کاملا متصل است
x = layers.Dense(128 * 64 * 64)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((64, 64, 128))(x)

# اضافه کردن چند لایه کانولوشنی
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# اپ‌سمپلینگ به ابعاد تصویر اصلی
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# تولید تصویر با ابعاد و کانال‌های مورد نظر
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = tf.keras.models.Model(generator_input, x)
generator.summary()

# ساختار دیسکریمیناتور
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

# یک لایه دراپ‌اوت برای جلوگیری از بیش‌برازش
x = layers.Dropout(0.4)(x)

# تصمیم‌گیری (واقعی یا جعلی)
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = tf.keras.models.Model(discriminator_input, x)
discriminator.summary()

# مدل دیسکریمیناتور را با اپتیمایزر و تابع زیان مرتبط می‌کنیم
discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=0.0008)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# در این مرحله، وزن‌های دیسکریمیناتور را ثابت می‌کنیم
discriminator.trainable = False

# GAN را با هم قرار می‌دهیم
gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.models.Model(gan_input, gan_output)

gan_optimizer = tf.keras.optimizers.RMSprop(lr=0.0004)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

 
