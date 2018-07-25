import tensorflow as tf
mnist = tf.keras.datasets.mnist

print("loading data ...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("building model...")

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("training model...")

model.fit(x_train, y_train, epochs=10)

print("evaluate model...")
model.evaluate(x_test, y_test)

print("saving model ....")
model.save_weights('weights.h5', save_format='h5')
model.load_weights('weights.h5')
file = open("model.json", "w")
file.write(model.to_json())
file.close()

print("finish")

