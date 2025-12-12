"""
Bean Disease Classification Model Training Script
CPU-optimized training script for bean disease detection.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("Bean Disease Classification Model Training")
print("="*60)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("="*60)

# ============================================================================
# 1. Configure Dataset Paths and Parameters
# ============================================================================
BASE_DIR = Path('Classification')
TRAIN_DIR = BASE_DIR / 'training'
VAL_DIR = BASE_DIR / 'validation'
TEST_DIR = BASE_DIR / 'test'

# Model parameters (CPU-optimized)
IMG_SIZE = 224
BATCH_SIZE = 16  # Small batch size for CPU
EPOCHS = 30
LEARNING_RATE = 0.0001

# Get class names from directory structure
CLASS_NAMES = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
NUM_CLASSES = len(CLASS_NAMES)

print(f"\nClasses: {CLASS_NAMES}")
print(f"Number of classes: {NUM_CLASSES}")

# Count images
def count_images(directory):
    """Count total images in a directory."""
    count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        count += len(list(directory.rglob(ext)))
    return count

train_count = count_images(TRAIN_DIR)
val_count = count_images(VAL_DIR)
test_count = count_images(TEST_DIR)

print(f"\nTraining images: {train_count}")
print(f"Validation images: {val_count}")
print(f"Test images: {test_count}")

# ============================================================================
# 2. Create Data Generators
# ============================================================================
print("\n" + "="*60)
print("Creating data generators...")
print("="*60)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

print(f"\nClass indices: {train_generator.class_indices}")
print(f"Training batches per epoch: {len(train_generator)}")
print(f"Validation batches: {len(val_generator)}")

# ============================================================================
# 3. Build the Model
# ============================================================================
print("\n" + "="*60)
print("Building model...")
print("="*60)

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    alpha=1.0
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# 4. Setup Callbacks
# ============================================================================
os.makedirs('models', exist_ok=True)

callbacks = [
    ModelCheckpoint(
        'models/bean_disease_best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================================
# 5. Phase 1: Feature Extraction Training
# ============================================================================
print("\n" + "="*60)
print("Phase 1: Feature Extraction Training")
print("="*60)
print("Training with frozen base model...")
print("This may take a while on CPU. Please be patient...\n")

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=callbacks,
    verbose=1
)

print("\nPhase 1 training completed!")

# ============================================================================
# 6. Phase 2: Fine-tuning
# ============================================================================
print("\n" + "="*60)
print("Phase 2: Fine-tuning")
print("="*60)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"Trainable parameters after unfreezing: {trainable_params:,}\n")

history_finetune = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=callbacks,
    verbose=1
)

print("\nFine-tuning completed!")

# ============================================================================
# 7. Visualize Training History
# ============================================================================
print("\n" + "="*60)
print("Generating training visualizations...")
print("="*60)

def combine_histories(hist1, hist2):
    """Combine two training history objects into one."""
    combined = {}
    for key in hist1.history.keys():
        combined[key] = hist1.history[key] + hist2.history[key]
    return combined

combined_history = combine_histories(history, history_finetune)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(combined_history['accuracy'], label='Training Accuracy', marker='o')
axes[0].plot(combined_history['val_accuracy'], label='Validation Accuracy', marker='s')
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(combined_history['loss'], label='Training Loss', marker='o')
axes[1].plot(combined_history['val_loss'], label='Validation Loss', marker='s')
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Final Training Accuracy: {combined_history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {combined_history['val_accuracy'][-1]:.4f}")

# ============================================================================
# 8. Evaluate on Test Set
# ============================================================================
print("\n" + "="*60)
print("Evaluating on test set...")
print("="*60)

best_model = keras.models.load_model('models/bean_disease_best_model.h5')
test_results = best_model.evaluate(test_generator, steps=len(test_generator), verbose=1)

print(f"\nTest Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
if len(test_results) > 2:
    print(f"Test Top-K Accuracy: {test_results[2]:.4f}")

# ============================================================================
# 9. Generate Classification Report and Confusion Matrix
# ============================================================================
print("\n" + "="*60)
print("Generating classification report...")
print("="*60)

test_generator.reset()
predictions = best_model.predict(test_generator, steps=len(test_generator), verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

class_names = list(test_generator.class_indices.keys())
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(true_classes, predicted_classes, target_names=class_names))

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 10. Save Final Model and Class Mappings
# ============================================================================
print("\n" + "="*60)
print("Saving model and metadata...")
print("="*60)

final_model_path = 'models/bean_disease_final_model.h5'
best_model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")

class_mapping = {
    'class_indices': {k: int(v) for k, v in test_generator.class_indices.items()},
    'class_names': class_names,
    'num_classes': NUM_CLASSES,
    'img_size': IMG_SIZE
}

with open('models/class_mapping.json', 'w') as f:
    json.dump(class_mapping, f, indent=2)

print(f"Class mapping saved to: models/class_mapping.json")
print(f"\nClass mapping:")
for class_name, idx in test_generator.class_indices.items():
    print(f"  {class_name}: {idx}")

print("\n" + "="*60)
print("Training Complete! (success)")
print("="*60)
print("\nFiles created:")
print("- models/bean_disease_best_model.h5")
print("- models/bean_disease_final_model.h5")
print("- models/class_mapping.json")
print("- models/training_history.png")
print("- models/confusion_matrix.png")

