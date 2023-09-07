from data_preparation import prepare_data
from cnn_model import create_cnn_model
from train_model import train_and_evaluate

def main():
    # Daten laden und vorbereiten
    train_images, train_labels, val_images, val_labels, test_images, test_labels, datagen = prepare_data()

    # CNN-Modell erstellen
    model = create_cnn_model(input_shape=(32, 32, 3))

    # Modell trainieren und bewerten
    train_and_evaluate(model, train_images, train_labels, val_images, val_labels, test_images, test_labels, datagen)


if __name__ == "__main__":
    main()