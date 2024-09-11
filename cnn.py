import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def _preprocess_mnist() -> tuple:
    '''
    Load data and normalize pixel values. Return a four-tuple in form x_train, y_train, x_test, y_test
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize to a 28 x 28 x 1 array (for the one color channel), with pixel value from 0-1
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

    # one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def _build_cnn() -> Sequential:
    '''
    Build a Sequential model with two Convolutional layers.
    '''
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_cnn():
    '''
    Train a Sequential model, save model locally.
    '''
    x_train, y_train, x_test, y_test = _preprocess_mnist()
    model = _build_cnn()

    model.fit(x_train, y_train, epochs=10, batch_size=200, validation_data=(x_test, y_test))
    model.save('mnist_cnn_model.keras')



if __name__ == "__main__":
    train_cnn()
