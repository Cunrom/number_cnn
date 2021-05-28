import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from PIL import Image
from tensorflow.keras.models import save_model, load_model
import numpy as np

img = np.invert(Image.open('./numbers/number.png').convert('L')).ravel()
img_tensor = tf.constant([img], dtype='float32', shape=(1, 784)) / 255.0
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Functional Model

inputs = layers.Input(shape=784)
x = layers.Dense(416, activation='relu')(inputs)
x = layers.Dense(208, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

filepath = './saved_model'

model = load_model(filepath, compile=True)
# model.fit(x_train, y_train, batch_size=160, epochs=500, verbose=1)
# save_model(model, filepath)

# print("Test Result:")
# model.evaluate(x_test, y_test, batch_size=100, verbose=1)

WIDTH, HEIGHT = 28, 28
data = img
data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
for row in data:
    print(' '.join('{:3}'.format(value) for value in row))

prediction = model.predict(img_tensor)
number_prediction = np.argmax(prediction)
print("The predicted number is: " + str(number_prediction))


