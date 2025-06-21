"""
cnn_vit.py

Image classification using Vision Transformer (ViT) feature extractor.
Applies transfer learning on breast lesion images using TensorFlow Hub ViT model.
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
    Resize and normalize images for ViT input.
    """
    x = tf.image.resize(x, (224, 224))
    return x / 255.0


def load_dataframe_splits(base_path):
    """
    Load and convert dataset CSVs to binary class labels.
    
    Args:
        base_path (str): Directory containing train/val/test CSV files.

    Returns:
        Tuple of train, validation, and test DataFrames.
    """
    def convert(df):
        df_copy = df.copy()
        df_copy['BINARY_CLASS'] = df_copy['CLASS'].apply(lambda x: 'NORM' if x == 'NORM' else 'HASTA')
        return df_copy

    train_df = convert(pd.read_csv(os.path.join(base_path, "train_pure.csv")))
    val_df = convert(pd.read_csv(os.path.join(base_path, "val.csv")))
    test_df = convert(pd.read_csv(os.path.join(base_path, "test.csv")))

    return train_df, val_df, test_df


def create_generators(train_df, val_df, test_df, base_dir):
    """
    Construct image generators with augmentation for training and preprocessing for validation/testing.

    Args:
        train_df, val_df, test_df (pd.DataFrame): DataFrames with image paths and labels.
        base_dir (str): Path prefix where image folders exist.

    Returns:
        Tuple of train, validation, and test generators.
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

    train_gen = build_generator(train_df, "train_pure", shuffle=True)
    val_gen = build_generator(val_df, "val", shuffle=False)
    test_gen = build_generator(test_df, "test", batch_size=16, shuffle=False)

    return train_gen, val_gen, test_gen


def build_model():
    """
    Create a sequential model using a frozen ViT encoder and a small classifier head.

    Returns:
        Compiled Keras model.
    """
    vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
    vit_layer = hub.KerasLayer(vit_url, trainable=False)

    model = models.Sequential([
        vit_layer,
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_metrics(history):
    """
    Plot training and validation accuracy/loss.

    Args:
        history: Keras training history object.
    """
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
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


def evaluate_model(model_path, test_generator, class_names):
    """
    Evaluate a trained model and display classification report and confusion matrix.

    Args:
        model_path (str): Path to saved Keras model.
        test_generator: Generator for test images.
        class_names (list): List of class labels.
    """
    model = load_model(model_path)
    loss, acc = model.evaluate(test_generator)
    print(f"\nTest Accuracy (Binary): {acc:.4f}")

    predictions = model.predict(test_generator)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Binary)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    # === Load dataset and prepare generators ===
    csv_base = "/home/senanur/deep_learning_homework/dataset/dataset_split"
    image_base = "/home/senanur/deep_learning_homework/dataset/dataset_split"

    train_df, val_df, test_df = load_dataframe_splits(csv_base)
    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df, image_base)

    # === Build model and calculate class weights ===
    model = build_model()
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(train_gen.classes),
                                         y=train_gen.classes)
    class_weights = dict(enumerate(class_weights))

    # === Callbacks for training ===
    callback_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        callbacks.ModelCheckpoint('best_vit_model.h5', monitor='val_accuracy', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    ]

    # === Train ===
    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        callbacks=callback_list,
        class_weight=class_weights
    )

    # === Evaluate ===
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    evaluate_model("best_vit_model.h5", test_gen, class_names)
    plot_metrics(history)


if __name__ == "__main__":
    main()
