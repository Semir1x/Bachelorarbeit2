import tensorflow as tf
from sklearn.model_selection import train_test_split


from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def prepare_data():
    # Daten laden
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalisierung der Pixelwerte
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # One-Hot-Encoding der Labels
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Aufteilen der Trainingsdaten in Trainings- und Validierungsdaten
    val_images = train_images[:5000]
    val_labels = train_labels[:5000]
    train_images = train_images[5000:]
    train_labels = train_labels[5000:]

    # Datenvermehrung
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )

    datagen.fit(train_images)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels, datagen




if __name__ == "__main__":
    train_images, train_labels, val_images, val_labels, test_images, test_labels = prepare_data()
    print(f"Trainingsdaten: {train_images.shape}, {train_labels.shape}")
    print(f"Validierungsdaten: {val_images.shape}, {val_labels.shape}")
    print(f"Testdaten: {test_images.shape}, {test_labels.shape}")