"""
MobileNetV3 Classifier for Malaria Parasite Detection
Trains a binary classifier to distinguish parasites from artifacts_bg

Dataset structure expected:
  dataset_root/
    train/
      parasites/
      artifacts_bg/
    val/
      parasites/
      artifacts_bg/
"""

import os
import shutil
import zipfile
from pathlib import Path
import argparse
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


class MobileNetV3Classifier:
    def __init__(self, 
                 dataset_root=None,
                 zip_path=None,
                 img_size=224,
                 batch_size=32,
                 learning_rate=0.001,
                 output_dir=None,
                 resize_mode='pad',
                 drive_root=None):
        """
        Initialize classifier.
        
        Args:
            dataset_root: Path to dataset root with train/val subdirs
            zip_path: Path to zip file to extract (if None, uses dataset_root)
            img_size: Image size for MobileNetV3 (default 224)
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            output_dir: Directory to save weights and results
            resize_mode: 'stretch' to distort, 'pad' to preserve aspect ratio (default 'pad')
            drive_root: Google Drive root path (Colab only). Defaults to /content/drive/MyDrive
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.resize_mode = resize_mode
        self.drive_root = drive_root
        
        if resize_mode not in ['stretch', 'pad']:
            raise ValueError("resize_mode must be 'stretch' or 'pad'")
        
        print(f"Using resize mode: {resize_mode}")
        
        # Setup Drive root for Colab
        if self._is_colab() and not self.drive_root:
            self.drive_root = "/content/drive/MyDrive/science_fair_2026/mbv3_classfier/crops_for_mobilenet"

        # Setup dataset
        if zip_path and os.path.exists(zip_path):
            self.dataset_root = self._extract_zip(zip_path)
        elif dataset_root:
            self.dataset_root = dataset_root
        else:
            # Try common locations
            self.dataset_root = self._find_dataset()

        # If running in Colab, copy dataset to local runtime for speed
        if self._is_colab():
            self.dataset_root = self._maybe_copy_dataset_to_local(self.dataset_root)
        
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_root}")
        
        # Setup output directory
        if output_dir:
            self.output_dir = output_dir
        elif self._is_colab():
            self.output_dir = f"{self.drive_root}/science_fair_2026/mbv3_classfier/results_classifier"
        else:
            self.output_dir = "./malaria_mobilenetv3"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model = None
        self.history = None
        
        print(f"Dataset root: {self.dataset_root}")
        print(f"Output directory: {self.output_dir}")
        print(f"Image dimensions: {self.img_size}×{self.img_size}")
    
    def _find_dataset(self):
        """Search for dataset in common locations."""
        common_paths = [
            os.path.expanduser("~/Downloads/crops_for_mobilenet"),
            "/tmp/crops_for_mobilenet",
            "./crops_for_mobilenet",
            "/content/crops_for_mobilenet"
        ]

        if self.drive_root:
            common_paths.extend([
                f"{self.drive_root}/Downloads/crops_for_mobilenet",
                f"{self.drive_root}/crops_for_mobilenet",
                f"{self.drive_root}/science_fair_2026/datasets/crops_for_mobilenet"
            ])
        
        for path in common_paths:
            if os.path.exists(path) and os.path.isdir(path):
                print(f"Found dataset at: {path}")
                return path
        
        raise FileNotFoundError(
            f"Dataset not found. Searched: {common_paths}\n"
            "Please provide dataset_root or zip_path parameter."
        )

    def _is_colab(self):
        try:
            import google.colab  # noqa: F401
            return True
        except Exception:
            return False
    
    def _extract_zip(self, zip_path):
        """Extract zip file and return extraction path."""
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        
        if self._is_colab() and zip_path.startswith("/content/drive/"):
            extract_dir = "/tmp/crops_for_mobilenet"
        else:
            extract_dir = os.path.splitext(zip_path)[0]
        
        if not os.path.exists(extract_dir):
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Extracted to {extract_dir}")
        else:
            print(f"Dataset already extracted at {extract_dir}")
        
        return extract_dir

    def _maybe_copy_dataset_to_local(self, dataset_root):
        """Copy dataset to /tmp on Colab for faster I/O."""
        if not dataset_root or not dataset_root.startswith("/content/drive/"):
            return dataset_root

        local_root = "/tmp/crops_for_mobilenet"
        if os.path.exists(local_root):
            print(f"Using cached local dataset at {local_root}")
            return local_root

        print(f"Copying dataset to local runtime: {local_root}")
        shutil.copytree(dataset_root, local_root)
        return local_root
    
    def _resize_with_padding(self, image):
        """
        Resize image to target size while preserving aspect ratio and quality.
        For small crops (<50px), limits upsampling to 1.5x to preserve detail.
        """
        h = tf.cast(tf.shape(image)[0], tf.float32)
        w = tf.cast(tf.shape(image)[1], tf.float32)
        target = tf.cast(self.img_size, tf.float32)
        min_dim = tf.minimum(h, w)
        
        # Calculate scale to fit within target size
        scale = tf.minimum(target / h, target / w)
        
        # For very small crops, limit upsampling to preserve quality
        # If original is <50px, don't upsample more than 1.5x
        scale = tf.cond(
            min_dim < 50.0,
            lambda: tf.minimum(scale, 1.5),  # Cap at 1.5x for tiny crops
            lambda: scale  # Use full scale for larger crops
        )
        
        new_h = tf.cast(h * scale, tf.int32)
        new_w = tf.cast(w * scale, tf.int32)
        
        # Resize preserving aspect ratio
        resized = tf.image.resize(image, [new_h, new_w])
        
        # Pad to target size with black borders (center the content)
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        
        padded = tf.image.pad_to_bounding_box(
            resized, pad_h, pad_w, self.img_size, self.img_size
        )
        return padded
    
    def _resize_with_stretch(self, image):
        """Resize image to target size by stretching (distorting if necessary)."""
        return tf.image.resize(image, [self.img_size, self.img_size])
    
    def _count_images(self):
        """Count images in dataset."""
        stats = {}
        for split in ['train', 'val']:
            split_path = os.path.join(self.dataset_root, split)
            if os.path.exists(split_path):
                parasites = len(os.listdir(os.path.join(split_path, 'parasites')))
                artifacts = len(os.listdir(os.path.join(split_path, 'artifacts_bg')))
                stats[split] = {'parasites': parasites, 'artifacts_bg': artifacts}
                print(f"{split.upper()}: {parasites} parasites, {artifacts} artifacts_bg")
        return stats
    
    def build_data_generators(self, augmentation=True):
        """Build ImageDataGenerators for train and validation."""
        # Choose target size based on resize mode
        target_size = (self.img_size, self.img_size) if self.resize_mode == 'stretch' else (None, None)
        
        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # For padding mode, set target_size to None and apply custom preprocessing
        if self.resize_mode == 'pad':
            # Custom function to apply padding
            def preprocess_with_pad(x):
                # x is a single image array in range [0, 255]
                x = tf.convert_to_tensor(x, dtype=tf.float32)
                return self._resize_with_padding(x) / 255.0
            
            preprocessing_function = preprocess_with_pad
            target_size_train = (None, None)
        else:
            preprocessing_function = None
            target_size_train = (self.img_size, self.img_size)
        
        # Load train data
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_root, 'train'),
            target_size=target_size_train,
            batch_size=self.batch_size,
            class_mode='binary',  # parasites=1, artifacts_bg=0
            classes={'artifacts_bg': 0, 'parasites': 1},
            shuffle=True,
            preprocessing_function=preprocessing_function
        )
        
        # Load validation data
        val_generator = val_datagen.flow_from_directory(
            os.path.join(self.dataset_root, 'val'),
            target_size=target_size_train,
            batch_size=self.batch_size,
            class_mode='binary',
            classes={'artifacts_bg': 0, 'parasites': 1},
            shuffle=False,
            preprocessing_function=preprocessing_function
        )
        
        return train_generator, val_generator
    
    def build_model(self, pretrained=True):
        """Build MobileNetV3 model."""
        if pretrained:
            base_model = MobileNetV3Large(
                input_shape=(self.img_size, self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
            print("Loaded MobileNetV3Large with ImageNet weights")
        else:
            base_model = MobileNetV3Large(
                input_shape=(self.img_size, self.img_size, 3),
                include_top=False,
                weights=None
            )
            print("Built MobileNetV3Large without pretrained weights")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build custom head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        model.summary()
        return model
    
    def train(self, 
              train_generator, 
              val_generator,
              epochs=50,
              unfreeze_at_epoch=20,
              early_stopping=True):
        """
        Train the model with optional unfreezing of base model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs to train
            unfreeze_at_epoch: Epoch at which to unfreeze base model
            early_stopping: Whether to use early stopping
        """
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.output_dir, 'best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.output_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        if early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            )
        
        # Add callback to unfreeze base model at specified epoch
        if unfreeze_at_epoch > 0:
            class UnfreezeCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if epoch == unfreeze_at_epoch - 1:
                        self.model.layers[0].trainable = True
                        print(f"\nUnfreezing base model at epoch {epoch + 1}")
                        self.model.compile(
                            optimizer=keras.optimizers.Adam(
                                learning_rate=self.learning_rate / 10  # Lower LR for fine-tuning
                            ),
                            loss='binary_crossentropy',
                            metrics=['accuracy']
                        )
            
            callbacks.append(UnfreezeCallback())
        
        # Calculate steps
        train_steps = max(1, len(train_generator))
        val_steps = max(1, len(val_generator))
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Train steps per epoch: {train_steps}")
        print(f"Val steps per epoch: {val_steps}")
        
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            steps_per_epoch=train_steps,
            validation_steps=val_steps
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training and validation curves."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=100)
        print(f"Saved training history plot to {plot_path}")
        plt.close()
    
    def save_results(self):
        """Save model, weights, and results summary."""
        # Save full model
        model_path = os.path.join(self.output_dir, 'model.h5')
        self.model.save(model_path)
        print(f"Saved model to {model_path}")
        
        # Save weights
        weights_path = os.path.join(self.output_dir, 'weights.h5')
        self.model.save_weights(weights_path)
        print(f"Saved weights to {weights_path}")
        
        # Save training summary
        if self.history:
            summary = {
                'final_train_accuracy': float(self.history.history['accuracy'][-1]),
                'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
                'final_train_loss': float(self.history.history['loss'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1]),
                'epochs_trained': len(self.history.history['accuracy']),
                'timestamp': datetime.now().isoformat()
            }
            
            summary_path = os.path.join(self.output_dir, 'training_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved training summary to {summary_path}")
            print(f"\nFinal Results:")
            print(f"  Train Accuracy: {summary['final_train_accuracy']:.4f}")
            print(f"  Val Accuracy: {summary['final_val_accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train MobileNetV3 on malaria parasite dataset')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='Path to dataset root directory')
    parser.add_argument('--zip-path', type=str, default=None,
                        help='Path to zip file containing dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for weights and results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size (default 224 for MobileNetV3)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Train from scratch instead of using pretrained weights')
    parser.add_argument('--unfreeze-epoch', type=int, default=20,
                        help='Epoch at which to unfreeze base model (0 to disable)')
    parser.add_argument('--resize-mode', type=str, choices=['stretch', 'pad'], default='pad',
                        help='How to resize images: stretch (distort) or pad (preserve aspect ratio)')
    parser.add_argument('--drive-root', type=str, default=None,
                        help='Google Drive root path (Colab). Example: /content/drive/MyDrive')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = MobileNetV3Classifier(
        dataset_root=args.dataset_root,
        zip_path=args.zip_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        resize_mode=args.resize_mode,
        drive_root=args.drive_root
    )
    
    # Print dataset info
    print("\nDataset Statistics:")
    classifier._count_images()
    
    # Build data generators
    print("\nBuilding data generators...")
    train_gen, val_gen = classifier.build_data_generators(
        augmentation=not args.no_augmentation
    )
    
    # Build model
    print("\nBuilding model...")
    classifier.build_model(pretrained=not args.no_pretrained)
    
    # Train
    print("\nTraining model...")
    classifier.train(
        train_gen, 
        val_gen,
        epochs=args.epochs,
        unfreeze_at_epoch=args.unfreeze_epoch
    )
    
    # Plot results
    print("\nGenerating plots...")
    classifier.plot_training_history()
    
    # Save results
    print("\nSaving results...")
    classifier.save_results()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
