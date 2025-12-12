"""
Bean Disease Classification Model Training Script
CPU-optimized training script for bean disease detection
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
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
from PIL import Image
import json
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
print(f"Using CPU: {len(tf.config.list_physical_devices('GPU')) == 0}")
print("="*60)

# ============================================================================
# 1. Configure Dataset Paths and Parameters
# ============================================================================
print("\n[1/8] Configuring dataset paths and parameters...")

BASE_DIR = Path('Classification')
TRAIN_DIR = BASE_DIR / 'training'
VAL_DIR = BASE_DIR / 'validation'
TEST_DIR = BASE_DIR / 'test'

# Model parameters (CPU-optimized)
IMG_SIZE = 224  # Standard size for MobileNetV2
BATCH_SIZE = 16  # Small batch size for CPU (adjust if you have more RAM)
EPOCHS = 30  # Start with 30, can increase if needed
LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning

# Get class names from directory structure
CLASS_NAMES = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
NUM_CLASSES = len(CLASS_NAMES)

print(f"Classes: {CLASS_NAMES}")
print(f"Number of classes: {NUM_CLASSES}")

# Count images in each split
def count_images(directory):
    """Count total images in a directory (including subdirectories)"""
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
# 2. Create Data Generators with Augmentation
# ============================================================================
print("\n[2/8] Creating data generators with augmentation...")

# Data augmentation for training (helps model generalize better)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Random rotation up to 20 degrees
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Random shear transformation
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill pixels outside boundaries
)

# No augmentation for validation and test (only normalization)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators.
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

print(f"Class indices: {train_generator.class_indices}")
print(f"Training batches per epoch: {len(train_generator)}")
print(f"Validation batches: {len(val_generator)}")

# ============================================================================
# 3. Build the Model-(Transfer Learning with MobileNetV2)
# ============================================================================
print("\n[3/8] Building model with MobileNetV2...")

# Load pre-trained MobileNetV2 model (trained on ImageNet)
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,  # Don't include the classification head
    weights='imagenet',  # Use pre-trained ImageNet weights
    alpha=1.0  # Width multiplier (1.0 = full width, smaller = faster but less accurate)
)

# Freeze the base model initially (we'll unfreeze later for fine-tuning)
base_model.trainable = False

# Build the complete model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Convert feature maps to vectors
    layers.Dropout(0.5),  # Regularization to prevent overfitting
    layers.Dense(128, activation='relu'),  # Dense layer for feature learning
    layers.Dropout(0.3),  # Additional regularization
    layers.Dense(NUM_CLASSES, activation='softmax')  # Output layer (4 classes)
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

# Calculate model size
total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# 4. Define Callbacks for Training
# ============================================================================
print("\n[4/8] Setting up training callbacks...")

# Create directory for saving models
os.makedirs('models', exist_ok=True)

# Callbacks for training
callbacks = [
    # Save the best model based on validation accuracy
    ModelCheckpoint(
        'models/bean_disease_best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Stop training if validation accuracy doesn't improve for 5 epochs
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate if validation loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce LR by half
        patience=3,  # Wait 3 epochs
        min_lr=1e-7,  # Minimum learning rate
        verbose=1
    )
]

print("Callbacks configured:")
print("- ModelCheckpoint: Saves best model based on validation accuracy")
print("- EarlyStopping: Stops training if no improvement for 5 epochs")
print("- ReduceLROnPlateau: Reduces learning rate when validation loss plateaus")

# ============================================================================
# 5. Train the Model (Phase 1: Feature Extraction)
# ============================================================================
print("\n[5/8] Starting Phase 1: Feature Extraction Training")
print("Training with frozen base model (faster, good starting point)")
print(f"This may take a while on CPU. Please be patient...\n")

# Train the model
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
# 6. Fine-tuning (Phase 2: Unfreeze Base Model)
# ============================================================================
print("\n[6/8] Starting Phase 2: Fine-tuning")
print("Unfreezing base model for fine-tuning...")

base_model.trainable = True

# Freeze early layers, fine-tune later layers
for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),  # 10x smaller LR
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

# Count trainable parameters
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"Trainable parameters after unfreezing: {trainable_params:,}")
print("This will take longer but should improve accuracy...\n")

# Continue training with fine-tuning
history_finetune = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,  # Fewer epochs for fine-tuning
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=callbacks,
    verbose=1
)

print("\nFine-tuning completed!")

# ============================================================================
# 7. Visualize Training History
# ============================================================================
print("\n[7/8] Visualizing training history...")

# Combine histories from both phases
def combine_histories(hist1, hist2):
    """Combine two training histories"""
    combined = {}
    for key in hist1.history.keys():
        combined[key] = hist1.history[key] + hist2.history[key]
    return combined

combined_history = combine_histories(history, history_finetune)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot accuracy
axes[0].plot(combined_history['accuracy'], label='Training Accuracy', marker='o')
axes[0].plot(combined_history['val_accuracy'], label='Validation Accuracy', marker='s')
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot loss
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
# 8. Evaluate on Test Set and Generate Reports
# ============================================================================
print("\n[8/8] Evaluating model on test set...")

# Load the best model
print("Loading best model for evaluation...")
best_model = keras.models.load_model('models/bean_disease_best_model.h5')

# Evaluate on test set
print("Evaluating on test set...")
test_results = best_model.evaluate(test_generator, steps=len(test_generator), verbose=1)

print(f"\nTest Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
if len(test_results) > 2:
    print(f"Test Top-K Accuracy: {test_results[2]:.4f}")

# Get predictions on test set
print("Generating predictions on test set...")
test_generator.reset()
predictions = best_model.predict(test_generator, steps=len(test_generator), verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = test_generator.classes

# Classification report
class_names = list(test_generator.class_indices.keys())
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Save the final model
final_model_path = 'models/bean_disease_final_model.h5'
best_model.save(final_model_path)
print(f"\nFinal model saved to: {final_model_path}")

# Save class indices for later use (important for Streamlit app)
class_mapping = {
    'class_indices': test_generator.class_indices,
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
print("Training Complete! ✅")
print("="*60)
print("\nFiles Created:")
print("- models/bean_disease_best_model.h5 - Best model based on validation accuracy")
print("- models/bean_disease_final_model.h5 - Final trained model")
print("- models/class_mapping.json - Class indices and metadata (needed for Streamlit app)")
print("- models/training_history.png - Training curves")
print("- models/confusion_matrix.png - Confusion matrix visualization")
print("\nReady to build Streamlit interface!")

