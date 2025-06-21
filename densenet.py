"""
densenet.py

Binary classification of breast lesion images using DenseNet121 as a feature extractor.
This script applies transfer learning on the MIAS dataset to distinguish between normal (NORM)
and abnormal (HASTA) cases using a frozen DenseNet backbone.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


def load_binary_dataframe(base_path):
    """
    Load train/val/test CSVs and convert multi-class labels into binary labels (NORM vs HASTA).

    Args:
        base_path (str): Path to the folder containing 'train_pure.csv', 'val.csv', and 'test.csv'.

    Returns:
        Tuple[pd.DataFrame]: DataFrames for training, validation, and test sets.
    """
    def convert(df):
        df = df.copy()
        df['BINARY_CLASS'] = df['CLASS'].apply(lambda x: 'NORM' if x == 'NORM' else 'HASTA')
        return df

    train = convert(pd.read_csv(os.path.join(base_path, 'train_pure.csv')))
    val = convert(pd.read_csv(os.path.join(base_path, 'val.csv')))
    test = convert(pd.read_csv(os.path.join(base_path, 'test.csv')))

    return train, val, test


def build_data_generators(train_df, val_df, test_df, base_dir):
    """
    Create image generators for training (with augmentation), validation, and test.

    Args:
        train_df, val_df, test_df (pd.DataFrame): The label metadata with filenames.
        base_dir (str): Path to image directories ('train_pure/', 'val/', 'test/').

    Returns:
        Tuple: Keras data generators for training, validation, and testing.
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    def flow(df, subdir, batch_size=32, shuffle=True):
        return train_datagen.flow_from_dataframe(
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
        flow(train_df, "train_pure", shuffle=True),
        flow(val_df, "val", shuffle=False),
        flow(test_df, "test", shuffle=False)
    )


def build_densenet_model():
    """
    Load the DenseNet121 model as a frozen base and attach a custom classifier head.

    Returns:
        tf.keras.Model: A compiled binary classifier model.
    """
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze feature extractor

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification head
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_training_metrics(history):
    """
    Plot accuracy and loss curves for both training and validation.

    Args:
        history: Training history object returned by model.fit()
    """
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_model(model_path, test_gen, class_names):
    """
    Load the best model and evaluate it on the test set.
    Also print the classification report and confusion matrix.

    Args:
        model_path (str): Path to the saved best model (.h5).
        test_gen: Keras generator for the test set.
        class_names: Ordered list of class names (e.g., ['HASTA', 'NORM']).
    """
    model = load_model(model_path)
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    y_pred = (model.predict(test_gen) > 0.5).astype(int).flatten()
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
    Full training and evaluation pipeline:
    - Loads and preprocesses the dataset
    - Builds and trains the model with class weights
    - Evaluates the best model on the test set
    """
    print("TensorFlow version:", tf.__version__)
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))

    # === Load and preprocess dataset ===
    base_dir = "/home/senanur/deep_learning_homework/models/dataset_split"
    train_df, val_df, test_df = load_binary_dataframe(base_dir)
    train_gen, val_gen, test_gen = build_data_generators(train_df, val_df, test_df, base_dir)

    # === Compute class weights to address imbalance ===
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))
    print("Computed class weights:", class_weights)

    # === Build and compile the DenseNet model ===
    model = build_densenet_model()

    # === Define training callbacks ===
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        callbacks.ModelCheckpoint('best_model_densenet121_binary_weighted.h5', monitor='val_accuracy', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]

    # === Train the model ===
    print("\nStarting model training...")
    history = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=callbacks_list,
        class_weight=class_weights
    )
    print("Training complete.")

    # === Evaluate the best model on the test set ===
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    evaluate_model("best_model_densenet121_binary_weighted.h5", test_gen, class_names)

    # === Plot training curves ===
    plot_training_metrics(history)


if __name__ == "__main__":
    main()
