"""
resnet50.py

Binary classification of breast lesion images using a ResNet50-based model.
A pre-trained ResNet50 (ImageNet weights) is used as a frozen feature extractor,
and a custom dense classifier is trained on top of it.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


def load_binary_dataframe(base_path):
    """
    Load and preprocess the CSV files for training, validation, and test.

    Multi-class labels are converted into binary:
    'NORM' remains as 'NORM', all others are labeled as 'HASTA'.

    Args:
        base_path (str): Directory containing split CSVs.

    Returns:
        Tuple[pd.DataFrame]: train, val, and test DataFrames with binary labels.
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
    Prepare Keras ImageDataGenerators for training, validation, and testing.

    Args:
        train_df, val_df, test_df (pd.DataFrame): Data splits with filenames and binary labels.
        base_dir (str): Directory where images are located.

    Returns:
        Tuple: training, validation, and test generators.
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


def build_resnet_model():
    """
    Create a ResNet50-based binary classifier with frozen convolutional layers.

    Returns:
        A compiled tf.keras model.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_training_metrics(history):
    """
    Plot accuracy and loss metrics over training epochs.

    Args:
        history (tf.keras.callbacks.History): Training history object.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_model(model_path, test_gen, class_names):
    """
    Evaluate a trained model on the test set and display performance metrics.

    Args:
        model_path (str): Path to saved best model (.h5 file).
        test_gen: Keras generator for test set.
        class_names: List of class labels.
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
    Full training pipeline:
    - Load dataset and apply preprocessing
    - Initialize and train the ResNet-based model
    - Evaluate final model on test data
    """
    print("TensorFlow version:", tf.__version__)
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))

    # === Load and prepare data ===
    base_dir = "/home/senanur/deep_learning_homework/dataset/dataset_split"
    train_df, val_df, test_df = load_binary_dataframe(base_dir)
    train_gen, val_gen, test_gen = build_data_generators(train_df, val_df, test_df, base_dir)

    # === Compute class weights ===
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    # === Build the model ===
    model = build_resnet_model()

    # === Define callbacks ===
    cb_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        callbacks.ModelCheckpoint('best_model_binary_weighted.h5', monitor='val_accuracy', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]

    # === Train the model ===
    print("\nStarting model training...")
    history = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=cb_list,
        class_weight=class_weights
    )
    print("Training complete.")

    # === Evaluate final model ===
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    evaluate_model("best_model_binary_weighted.h5", test_gen, class_names)
    plot_training_metrics(history)


if __name__ == "__main__":
    main()
