  import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import random



# Data Directory (Update path accordingly)
IRMAS_PATH = '/content/IRMAS-TrainingData'

# Install required libraries
!pip install librosa

def pad_or_truncate(spec, target_shape=(128, 128)):
    """Ensure all spectrograms are of uniform size by padding or truncating."""
    pad_width = [(0, max(0, target_shape[0] - spec.shape[0])),
                 (0, max(0, target_shape[1] - spec.shape[1]))]
    spec = np.pad(spec, pad_width, mode='constant')
    return spec[:target_shape[0], :target_shape[1]]

def load_audio_files(directory, sr=22050, target_shape=(128, 128)):
    """Load audio files, convert to Mel spectrograms, and fix shape inconsistencies."""
    mel_specs = []
    labels = []
    class_map = {}
    class_id = 0

    for instrument in os.listdir(directory):
        instrument_path = os.path.join(directory, instrument)
        if os.path.isdir(instrument_path):
            class_map[instrument] = class_id
            for file in os.listdir(instrument_path):
                file_path = os.path.join(instrument_path, file)
                if file.endswith('.wav'):
                    y, _ = librosa.load(file_path, sr=sr)
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    mel_spec_db = pad_or_truncate(mel_spec_db, target_shape)

                    mel_specs.append(mel_spec_db)
                    labels.append(class_id)
            class_id += 1

    return np.array(mel_specs), np.array(labels), class_map

# Load data
mel_specs, labels, class_map = loadq_audio_files(IRMAS_PATH)

# Normalize data
mel_specs = mel_specs[..., np.newaxis]  # Add channel dimension
mel_specs = (mel_specs - np.min(mel_specs)) / (np.max(mel_specs) - np.min(mel_specs))  # Min-max scaling

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(mel_specs, labels, test_size=0.2, random_state=42, stratify=labels)

# Convert labels to categorical
num_classes = len(class_map)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# CNN Model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.0001), input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.0001)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create model
model = create_cnn_model(X_train.shape[1:], num_classes)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Train model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=60,
                    batch_size=32,
                    callbacks=[reduce_lr, early_stop])

# Save model
model.save("/content/drive/MyDrive/irmas_cnn_model.h5")

# Evaluate model
eval_result = model.evaluate(X_test, y_test)
print("Test Accuracy: {eval_result[1] * 100:.2f}%")

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels
y_true = np.argmax(y_test, axis=1)  # Convert true labels to class labels

conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_map.keys(), yticklabels=class_map.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Accuracy and Loss Plot
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()