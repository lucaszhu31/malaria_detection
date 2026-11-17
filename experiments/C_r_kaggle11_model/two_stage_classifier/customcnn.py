"""
Custom CNN Classifier for Low-Resolution Malaria Parasite Detection
Optimized for 30x30 to 115x115 pixel YOLO crops
Uses padding for small images and minimal downsampling for large images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import os
import shutil
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


class CustomCNNClassifier:
    """Custom lightweight CNN optimized for small image classification (30-115px)"""
    
    def __init__(self, img_size=96, data_dir=None, output_dir=None, 
                 batch_size=32, epochs=30, initial_lr=0.001):
        """
        Initialize Custom CNN Classifier
        
        Args:
            img_size: Target image size (default 96x96)
            data_dir: Path to dataset directory with train/val splits
            output_dir: Path to save results
            batch_size: Batch size for training
            epochs: Number of training epochs
            initial_lr: Initial learning rate
        """
        self.img_size = img_size
        self.data_dir = Path(data_dir) if data_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.model = None
        self.history = None
        
    def _resize_with_padding(self, img):
        """
        Resize image preserving quality:
        - Small images (<64px): Pad without upsampling to preserve sharpness
        - Large images (>=64px): Resize normally with minimal downsampling
        
        Args:
            img: Input image tensor
            
        Returns:
            Processed image tensor of shape (img_size, img_size, 3)
        """
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        max_dim = tf.maximum(h, w)
        
        # Small images: pad without upsampling to keep crisp pixels
        def pad_small():
            # Center image on canvas
            pad_h = self.img_size - h
            pad_w = self.img_size - w
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            padded = tf.pad(img, [[top, bottom], [left, right], [0, 0]], 
                          constant_values=0)
            # Cast to float32 and set explicit shape
            padded = tf.cast(padded, tf.float32)
            padded = tf.ensure_shape(padded, [self.img_size, self.img_size, 3])
            return padded
        
        # Large images: resize normally
        def resize_large():
            resized = tf.image.resize(img, [self.img_size, self.img_size], 
                                    method='bilinear')
            # Ensure explicit shape
            resized = tf.ensure_shape(resized, [self.img_size, self.img_size, 3])
            return resized
        
        # Choose processing based on size
        processed = tf.cond(
            max_dim < 64,
            pad_small,
            resize_large
        )
        
        return processed
    
    def _load_and_preprocess(self, file_path, label):
        """Load and preprocess a single image"""
        # Read image
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        
        # Apply padding/resizing strategy
        img = self._resize_with_padding(img)
        
        # Ensure correct shape
        img.set_shape([self.img_size, self.img_size, 3])
        
        # Normalize to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        
        return img, label
    
    def _augment_image(self, img, label):
        """Apply data augmentation"""
        # Random horizontal flip
        img = tf.image.random_flip_left_right(img)
        
        # Random vertical flip
        img = tf.image.random_flip_up_down(img)
        
        # Random rotation (90 degree increments)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img = tf.image.rot90(img, k=k)
        
        # Random brightness adjustment
        img = tf.image.random_brightness(img, 0.2)
        
        # Random contrast adjustment
        img = tf.image.random_contrast(img, 0.8, 1.2)
        
        # Clip to valid range
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img, label
    
    def _list_files_and_labels(self, split_dir):
        """Get all image files and their labels from directory"""
        class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        label_map = {name: idx for idx, name in enumerate(class_names)}
        
        files = []
        labels = []
        
        for class_name in class_names:
            class_dir = split_dir / class_name
            class_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + \
                         list(class_dir.glob('*.png'))
            
            files.extend([str(f) for f in class_files])
            labels.extend([label_map[class_name]] * len(class_files))
        
        return files, labels, class_names
    
    def _build_tf_dataset(self, split_dir, is_training=True):
        """Build tf.data pipeline for efficient data loading"""
        files, labels, class_names = self._list_files_and_labels(split_dir)
        
        print(f"\n{'Training' if is_training else 'Validation'} set:")
        print(f"  Total images: {len(files)}")
        for i, class_name in enumerate(class_names):
            count = labels.count(i)
            print(f"  {class_name}: {count}")
        
        # Create dataset from file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((files, labels))
        
        # Shuffle if training
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(files), reshuffle_each_iteration=True)
        
        # Load and preprocess
        dataset = dataset.map(self._load_and_preprocess, 
                            num_parallel_calls=tf.data.AUTOTUNE)
        
        # Augment if training
        if is_training:
            dataset = dataset.map(self._augment_image, 
                                num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset, len(files)
    
    def build_datasets(self):
        """Build training and validation datasets"""
        train_dir = self.data_dir / 'train'
        val_dir = self.data_dir / 'val'
        
        self.train_dataset, self.train_samples = self._build_tf_dataset(train_dir, is_training=True)
        self.val_dataset, self.val_samples = self._build_tf_dataset(val_dir, is_training=False)
        
        self.steps_per_epoch = self.train_samples // self.batch_size
        self.validation_steps = self.val_samples // self.batch_size
        
        print(f"\nDataset ready:")
        print(f"  Steps per epoch: {self.steps_per_epoch}")
        print(f"  Validation steps: {self.validation_steps}")
        
        return self.train_dataset, self.val_dataset
    
    def build_model(self):
        """
        Build custom CNN optimized for low-resolution images (30-115px)
        
        Architecture:
        - 4 convolutional blocks with gradual channel growth (32→64→128→256)
        - Small 3x3 kernels (5x5 only at start) for appropriate receptive field
        - BatchNorm for stable training
        - Dropout for regularization
        - ~180K parameters (vs MobileNetV3's 5.4M)
        """
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        # BLOCK 1: Initial Feature Extraction (96→48)
        x = layers.Conv2D(32, (5, 5), padding='same', name='block1_conv1')(inputs)
        x = layers.BatchNormalization(name='block1_bn1')(x)
        x = layers.ReLU(name='block1_relu1')(x)
        
        x = layers.Conv2D(32, (3, 3), padding='same', name='block1_conv2')(x)
        x = layers.BatchNormalization(name='block1_bn2')(x)
        x = layers.ReLU(name='block1_relu2')(x)
        
        x = layers.MaxPooling2D((2, 2), name='block1_pool')(x)
        
        # BLOCK 2: Edge & Shape Detection (48→24)
        x = layers.Conv2D(64, (3, 3), padding='same', name='block2_conv1')(x)
        x = layers.BatchNormalization(name='block2_bn1')(x)
        x = layers.ReLU(name='block2_relu1')(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', name='block2_conv2')(x)
        x = layers.BatchNormalization(name='block2_bn2')(x)
        x = layers.ReLU(name='block2_relu2')(x)
        
        x = layers.MaxPooling2D((2, 2), name='block2_pool')(x)
        
        # BLOCK 3: Pattern Recognition - Ring Structures (24→12)
        x = layers.Conv2D(128, (3, 3), padding='same', name='block3_conv1')(x)
        x = layers.BatchNormalization(name='block3_bn1')(x)
        x = layers.ReLU(name='block3_relu1')(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)
        x = layers.BatchNormalization(name='block3_bn2')(x)
        x = layers.ReLU(name='block3_relu2')(x)
        
        x = layers.MaxPooling2D((2, 2), name='block3_pool')(x)
        
        # BLOCK 4: High-Level Features (12→6)
        x = layers.Conv2D(256, (3, 3), padding='same', name='block4_conv1')(x)
        x = layers.BatchNormalization(name='block4_bn1')(x)
        x = layers.ReLU(name='block4_relu1')(x)
        
        x = layers.Dropout(0.3, name='block4_dropout')(x)
        x = layers.MaxPooling2D((2, 2), name='block4_pool')(x)
        
        # Classification Head
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        
        x = layers.Dense(256, name='fc1')(x)
        x = layers.BatchNormalization(name='fc1_bn')(x)
        x = layers.ReLU(name='fc1_relu')(x)
        x = layers.Dropout(0.4, name='fc1_dropout')(x)
        
        x = layers.Dense(128, name='fc2')(x)
        x = layers.BatchNormalization(name='fc2_bn')(x)
        x = layers.ReLU(name='fc2_relu')(x)
        x = layers.Dropout(0.3, name='fc2_dropout')(x)
        
        outputs = layers.Dense(1, activation='sigmoid', name='predictions')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='CustomCNN')
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        return self.model
    
    def compile_model(self):
        """Compile model with optimizer and loss"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.initial_lr),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
            ]
        )
        print(f"\nModel compiled with learning rate: {self.initial_lr}")
    
    def calculate_class_weights(self):
        """Calculate class weights to handle imbalance (2:1 parasite:artifact ratio)"""
        # Get class distribution from training directory
        train_dir = self.data_dir / 'train'
        _, labels, class_names = self._list_files_and_labels(train_dir)
        
        # Count samples per class
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        # Calculate weights (inverse frequency)
        class_weights = {}
        for cls, count in zip(unique, counts):
            weight = total / (len(unique) * count)
            class_weights[cls] = weight
            print(f"  Class {class_names[cls]} ({cls}): {count} samples, weight: {weight:.2f}")
        
        return class_weights
    
    def train(self):
        """Train the model"""
        # Calculate class weights for imbalanced dataset
        print("\nCalculating class weights:")
        class_weights = self.calculate_class_weights()
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpoint - save best model
        checkpoint_path = self.output_dir / 'best.keras'
        checkpoint = keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # TensorBoard logging
        tensorboard_dir = self.output_dir / 'logs'
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1
        )
        callbacks.append(tensorboard)
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Learning rate reduction
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Train model
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Class weights: {class_weights}")
        
        self.history = self.model.fit(
            self.train_dataset,
            epochs=self.epochs,
            validation_data=self.val_dataset,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return self.history
    
    def save_results(self):
        """Save model, weights, and training history"""
        print("\nSaving results...")
        
        # Save final model
        model_path = self.output_dir / 'final_model.keras'
        self.model.save(str(model_path))
        print(f"  Final model: {model_path}")
        
        # Save weights only
        weights_path = self.output_dir / 'final_model.weights.h5'
        self.model.save_weights(str(weights_path))
        print(f"  Final weights: {weights_path}")
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        history_dict = {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"  Training history: {history_path}")
        
        # Save training summary
        summary_path = self.output_dir / 'training_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Custom CNN Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Image size: {self.img_size}x{self.img_size}\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Epochs: {self.epochs}\n")
            f.write(f"Initial LR: {self.initial_lr}\n\n")
            
            f.write("Final Metrics:\n")
            for metric, values in self.history.history.items():
                f.write(f"  {metric}: {values[-1]:.4f}\n")
            
            f.write("\nBest Metrics:\n")
            best_epoch = np.argmax(self.history.history['val_accuracy'])
            f.write(f"  Best epoch: {best_epoch + 1}\n")
            for metric, values in self.history.history.items():
                f.write(f"  {metric}: {values[best_epoch]:.4f}\n")
        
        print(f"  Training summary: {summary_path}")
        
        # Plot and save training curves
        self._plot_training_history()
    
    def _plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Val')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'training_history.png'
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        print(f"  Training plot: {plot_path}")
        plt.close()
    
    def _maybe_copy_dataset_to_local(self):
        """Copy dataset from Google Drive to local temp for faster I/O (Colab optimization)"""
        # Check if running on Colab
        try:
            import google.colab
            is_colab = True
        except:
            is_colab = False
        
        if not is_colab:
            return self.data_dir
        
        # Check if data_dir is on Drive
        if not str(self.data_dir).startswith('/content/drive'):
            return self.data_dir
        
        # Local copy path
        local_data_dir = Path('/tmp/crops_for_mobilenet')
        
        # Check if already copied
        if local_data_dir.exists():
            print(f"\nDataset already copied to {local_data_dir}")
            print("  Reusing existing copy for faster I/O")
            return local_data_dir
        
        print(f"\nCopying dataset from Drive to /tmp for faster I/O...")
        print(f"  Source: {self.data_dir}")
        print(f"  Destination: {local_data_dir}")
        
        shutil.copytree(self.data_dir, local_data_dir)
        print("  Copy complete!")
        
        return local_data_dir


def main():
    parser = argparse.ArgumentParser(description='Train Custom CNN for malaria parasite classification')
    
    # Paths
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory with train/val splits')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Path to save results')
    
    # Model parameters
    parser.add_argument('--img-size', type=int, default=96,
                       help='Target image size (default: 96)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    
    # Colab optimization
    parser.add_argument('--copy-to-tmp', action='store_true',
                       help='Copy dataset to /tmp for faster I/O on Colab')
    
    args, unknown = parser.parse_known_args()  # Ignore Jupyter kernel args
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize classifier
    classifier = CustomCNNClassifier(
        img_size=args.img_size,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        initial_lr=args.lr
    )
    
    # Maybe copy dataset to local storage (Colab optimization)
    if args.copy_to_tmp:
        classifier.data_dir = classifier._maybe_copy_dataset_to_local()
    
    # Build datasets
    print("\nBuilding datasets...")
    classifier.build_datasets()
    
    # Build model
    print("\nBuilding model...")
    classifier.build_model()
    
    # Compile model
    classifier.compile_model()
    
    # Train model
    classifier.train()
    
    # Save results
    classifier.save_results()
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
