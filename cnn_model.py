# cnn_model.py

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_cnn_model(input_shape):
    model = Sequential()

    # Erste Convolutional Layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))  # Dropout nach der ersten Convolutional Layer

    # Zweite Convolutional Layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))  # Dropout nach der zweiten Convolutional Layer

    # Dritte Convolutional Layer
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))  # Dropout vor der Ausgabeschicht

    # Ausgabeschicht
    model.add(Dense(10, activation='softmax'))

    return model


if __name__ == "__main__":
    model = create_cnn_model()
    model.summary()