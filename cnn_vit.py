"""
cnn_vit.py

Image classification of breast lesion images using a Vision Transformer (ViT) feature extractor.
This model uses a pre-trained ViT encoder from TensorFlow Hub and adds a lightweight classifier
for binary classification (NORM vs HASTA).
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


def vit_preprocess_input(x):
    """
    Preprocess input for ViT: resize to 224x224 and normalize pixel values to [0, 1].

    Args:
        x: Input image tensor.

    Returns:
        Preprocessed tensor.
    """
    x = tf.image.resize(x, (224, 224))
    return x / 255.0


def load_dataframe_splits(base_path):
    """
    Load train/val/test CSVs and convert multi-class labels into binary labels (NORM vs HASTA).

    Args:
        base_path (str): Directory containing 'train_pure.csv', 'val.csv', and 'test.csv'.

    Returns:
        Tuple of DataFrames: (train_df, val_df, test_df)
    """
    def convert(df):
        df_copy = df.copy()
        df_copy['BINARY_CLASS'] = df_copy['CLASS'].apply(lambda x: 'NORM' if x == 'NORM' else 'HASTA')
        return df_copy

    train = convert(pd.read_csv(os.path.join(base_path, "train_pure.csv")))
    val = convert(pd.read_csv(os.path.join(base_path, "val.csv")))
    test = convert(pd.read_csv(os.path.join(base_path, "test.csv")))

    return train, val, test


def create_generators(train_df, val_df, test_df, base_dir):
    """
    Create data generators for training, validation, and test using flow_from_dataframe.

    Args:
        train_df, val_df, test_df: DataFrames with image paths and binary labels.
        base_dir (str): Base directory for images.

    Returns:
        Tuple of Keras ImageDataGenerators: (train_gen, val_gen, test_gen)
    """
    train_aug = ImageDataGenerator(
        preprocessing_function=vit_preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_test_aug = ImageDataGenerator(preprocessing_function=vit_preprocess_input)

    def build_generator(df, subdir, batch_size=32, shuffle=True):
        return train_aug.flow_from_dataframe(
            dataframe=df,
            directory=os.path.join(base_dir, subdir),
            x_col="filename",
            y_col="BINARY_CLASS",
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="binary",
            shuffle=shuffle
        )

    return (
        build_generator(train_df, "train_pure", shuffle=True),
        build_generator(val_df, "val", shuffle=False),
        build_generator(test_df, "test", batch_size=16, shuffle=False)
    )


def build_vit_model():
    """
    Build a classifier model using frozen ViT backbone from TensorFlow Hub.

    Returns:
        Compiled Keras model ready for training.
    """
    vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
    vit_layer = hub.KerasLayer(vit_url, trainable=False)

    model = models.Sequential([
        vit_layer,                     # Output: (batch_size, 768)
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_metrics(history):
    """
    Plot training and validation accuracy and loss curves.

    Args:
        history: Keras training history object.
    """
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_model(model_path, test_gen, class_names):
    """
    Evaluate a trained model and display classification metrics.

    Args:
        model_path (str): Path to the best saved model (.h5 file).
        test_gen: Keras test data generator.
        class_names: List of class names corresponding to label indices.
    """
    model = load_model(model_path)
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    predictions = model.predict(test_gen)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    """
    Full training pipeline:
    - Loads data and preprocesses it
    - Builds ViT-based model
    - Trains with class weighting
    - Evaluates on test set
    """
    print("TensorFlow version:", tf.__version__)
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))

    base_dir = "/home/senanur/deep_learning_homework/dataset/dataset_split"
    train_df, val_df, test_df = load_dataframe_splits(base_dir)
    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df, base_dir)

    # === Compute class weights for imbalanced dataset ===
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))
    print("Class Weights:", class_weights)

    # === Build and compile model ===
    model = build_vit_model()

    # === Define training callbacks ===
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        callbacks.ModelCheckpoint('best_vit_model.h5', monitor='val_accuracy', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]

    # === Train the model ===
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        callbacks=callbacks_list,
        class_weight=class_weights
    )
    print("Training complete.")

    # === Evaluate on test set ===
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    evaluate_model("best_vit_model.h5", test_gen, class_names)
    plot_metrics(history)


if __name__ == "__main__":
    main()
