"""
FAST Training Script - Optimized for Speed
Uses MobileNetV3 (lighter model) instead of DenseNet for MUCH faster training
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import json
from datetime import datetime

# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[✓] GPU Available: {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"[!] GPU setup error: {e}")
else:
    print("[!] No GPU detected - using CPU")

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# ==================== CONFIG ====================
IMG_HEIGHT = 160
IMG_WIDTH = 160
BATCH_SIZE = 32
EPOCHS = 50              # SIGNIFICANTLY REDUCED
DATASET_PATH = 'dataset1'
MODEL_OUTPUT_DIR = 'dataset1_stability_model_fast'
LEARNING_RATE = 0.001
CLASSES = ['good', 'bad', 'medium']
NUM_CLASSES = 3

print(f"\n{'='*60}")
print(f"FAST TRAINING - MobileNetV3 (Lightweight Model)")
print(f"{'='*60}")
print(f"Image Size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Dataset: {DATASET_PATH}\n")
# ================================================

def verify_dataset():
    """Verify dataset"""
    print("[*] Verifying dataset...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path '{DATASET_PATH}' not found!")
    
    for class_name in CLASSES:
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_path):
            raise FileNotFoundError(f"Class folder '{class_name}' not found")
        
        images = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        print(f"  ✓ {class_name:8} - {len(images):4} images")

def create_generators():
    """Create data generators from dataset1 directly"""
    print("\n[*] Creating data generators...")
    
    # Light augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        validation_split=0.2,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        validation_split=0.2,
        seed=42
    )
    
    print(f"  ✓ Train samples: {train_generator.samples}")
    print(f"  ✓ Val samples: {val_generator.samples}")
    
    return train_generator, val_generator

def build_model():
    """Build FAST MobileNetV3Small model"""
    print("\n[*] Building MobileNetV3Small (FAST)...")
    
    base_model = MobileNetV3Small(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    # Minimal top layers - VERY FAST
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"  ✓ Model built - {model.count_params():,} parameters")
    return model, base_model

def train(model, base_model, train_gen, val_gen):
    """Train model - FAST"""
    print("\n[*] Training model (Phase 1: Frozen base)...\n")
    
    os.makedirs(os.path.join(MODEL_OUTPUT_DIR, 'checkpoints'), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_OUTPUT_DIR, 'checkpoints', 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Quick fine-tune
    print("\n[*] Fine-tuning (Phase 2: Last 10 layers unfrozen)...\n")
    
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    optimizer = Adam(learning_rate=LEARNING_RATE / 10)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate(model, train_gen, val_gen):
    """Evaluate model"""
    print("\n[*] Evaluating model...\n")
    
    val_steps = np.ceil(val_gen.samples / BATCH_SIZE)
    predictions = model.predict(val_gen, steps=val_steps, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    
    accuracy = accuracy_score(true_classes, pred_classes)
    print(f"  ✓ Validation Accuracy: {accuracy*100:.2f}%\n")
    
    class_names = list(val_gen.class_indices.keys())
    print(classification_report(true_classes, pred_classes, target_names=class_names))
    
    cm = confusion_matrix(true_classes, pred_classes)
    return accuracy, cm

def main():
    try:
        # Step 1: Verify
        verify_dataset()
        
        # Step 2: Create generators
        train_gen, val_gen = create_generators()
        
        # Step 3: Build model
        model, base_model = build_model()
        
        # Step 4: Train
        history = train(model, base_model, train_gen, val_gen)
        
        # Step 5: Load best and evaluate
        best_model_path = os.path.join(MODEL_OUTPUT_DIR, 'checkpoints', 'best_model.keras')
        model = keras.models.load_model(best_model_path)
        accuracy, cm = evaluate(model, train_gen, val_gen)
        
        # Step 6: Save final model
        final_model_path = os.path.join(MODEL_OUTPUT_DIR, 'final_model.keras')
        model.save(final_model_path)
        print(f"  ✓ Final model saved to {final_model_path}")
        
        print("\n" + "="*60)
        print(f"✓ TRAINING COMPLETED!")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Model: {MODEL_OUTPUT_DIR}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
