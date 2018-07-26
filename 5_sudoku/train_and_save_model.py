from keras.utils import np_utils
from models.cnnmodel import CNNModel
from keras.datasets import mnist

# load handwritten digits
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# copy from 4_ml/4_ml_2.py
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

# image size
img_rows, img_cols = 28, 28
# count of classes 10: 0...9
nb_classes = 10

# converts a class vector (list of labels in one vector (as for SVM)
# to binary class matrix (one-n-encoding)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# we need to reshape the input data to fit keras.io input matrix format
X_train, X_test = CNNModel.reshape_input_data(X_train, X_test)

# hyperparameter
nb_epoch = 1
batch_size = 128

model = CNNModel.load_model(nb_classes)
history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch, verbose=1,
                    validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# new: save model
weights_fn = "models/weights" + str(nb_epoch) + ".h5"
print("saving model to "+weights_fn)
model.save_weights(weights_fn)