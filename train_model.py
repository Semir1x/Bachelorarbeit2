# train_model.py

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

def train_and_evaluate(model, train_images, train_labels, val_images, val_labels, test_images, test_labels, datagen):
    # Kompilieren des Modells
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks definieren
    checkpoint = ModelCheckpoint('best_model.tf', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    # Training des Modells mit Datenvermehrung
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=64),
                        epochs=50,
                        validation_data=(val_images, val_labels),
                        callbacks=[checkpoint, early_stopping])

    # Evaluieren des Modells mit den Testdaten
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    return history



def plot_training_results(history):
    """
    Stellt die Trainings- und Validierungsergebnisse dar.

    Args:
        history: Trainingshistorie des Modells.
    """

    # Genauigkeit darstellen
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')

    # Verlust darstellen
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.show()


if __name__ == "__main__":
    # Dieser Block dient nur zu Testzwecken und kann angepasst werden.
    from data_preparation import load_and_prepare_data
    from cnn_model import create_cnn_model

    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_and_prepare_data()
    model = create_cnn_model()
    history = train_and_evaluate(model, train_images, train_labels, val_images, val_labels, test_images, test_labels)

    # Visualisieren Sie die Ergebnisse nach dem Training
    plot_training_results(history)
