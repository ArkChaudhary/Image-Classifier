"""
CIFAR-10 CNN Classifier Training Pipeline
This module contains the CIFAR10Classifier class with all training and evaluation functionality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, classification_report, 
                           roc_curve, roc_auc_score, precision_score, 
                           recall_score, f1_score)
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
import time
from datetime import datetime

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class CIFAR10Classifier:
    """
    A CNN classifier for CIFAR-10 dataset with comprehensive evaluation.
    
    This class handles the complete pipeline from data loading to model evaluation,
    including visualization and reporting capabilities.
    """
    
    def __init__(self):
        """Initialize the classifier with CIFAR-10 class names."""
        self.classes = ["airplane", "automobile", "bird", "cat", "deer", 
                       "dog", "frog", "horse", "ship", "truck"]
        self.model = None
        self.history = None
        self.training_time = 0
        
    def load_and_preprocess_data(self):
        """
        Load CIFAR-10 data and preprocess it.
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test) preprocessed data
        """
        print("Loading CIFAR-10 dataset...")
        (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
        
        # Reshape labels to 1D
        y_train = y_train.reshape(-1,)
        y_test = y_test.reshape(-1,)
        
        # Normalize pixel values to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Number of classes: {len(self.classes)}")
        
        return X_train, y_train, X_test, y_test
    
    def build_model(self):
        """
        Build the CNN model with batch normalization and dropout.
        
        Returns:
            tensorflow.keras.Model: Compiled CNN model
        """
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", 
                         input_shape=(32,32,3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Second convolutional block
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.3),
            
            # Third convolutional block
            layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Fully connected layers
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            
            # Output layer - 10 classes for CIFAR-10
            layers.Dense(10, activation="softmax")
        ])
        
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def create_data_generators(self, X_train, y_train):
        """
        Create data generators for augmentation.
        
        Args:
            X_train: Training images
            y_train: Training labels
            
        Returns:
            ImageDataGenerator: Configured data generator
        """
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        
        return datagen
    
    def train_model(self, X_train, y_train, epochs=20, batch_size=32):
        """
        Train the model with early stopping and checkpointing.
        
        Args:
            X_train: Training images
            y_train: Training labels
            epochs: Maximum number of epochs
            batch_size: Training batch size
        """
        print("Building model...")
        self.model = self.build_model()
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        print(f"\nTotal parameters: {self.model.count_params():,}")
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print(f"\nStarting training for up to {epochs} epochs...")
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_split=0.1,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        print(f"\nTraining completed in {self.training_time:.2f} seconds")
        
        # Save final model
        self.model.save('models/final_model.h5')
        
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation with multiple metrics.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            tuple: (y_pred, y_pred_proba) predictions and probabilities
        """
        print("\nEvaluating model...")
        
        # Basic evaluation
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Generate predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate macro metrics
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"\nMacro Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Generate and print classification report
        report = classification_report(y_test, y_pred, target_names=self.classes)
        print(f"\nDetailed Classification Report:")
        print(report)
        
        # Save classification report to file
        self._save_classification_report(test_accuracy, report)
        
        return y_pred, y_pred_proba
    
    def _save_classification_report(self, test_accuracy, report):
        """Save classification report to text file."""
        with open('reports/classification_report.txt', 'w') as f:
            f.write(f"CIFAR-10 Classification Report\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Training Time: {self.training_time:.2f} seconds\n")
            f.write(f"Model Parameters: {self.model.count_params():,}\n\n")
            f.write(report)
    
    def plot_learning_curves(self):
        """Plot training and validation learning curves."""
        if self.history is None:
            print("No training history available")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot normalized confusion matrix with class labels.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_class_metrics(self, y_true, y_pred):
        """
        Plot and save per-class precision, recall, and F1-score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        # Calculate per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Create DataFrame for easier handling
        metrics_df = pd.DataFrame({
            'Class': self.classes,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1-Score': f1_per_class
        })
        
        # Plot metrics
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(self.classes))
        width = 0.25
        
        ax.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.classes, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('plots/per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print and save metrics table
        print("\nPer-Class Metrics:")
        print(metrics_df.round(4))
        metrics_df.round(4).to_csv('reports/per_class_metrics.csv', index=False)
    
    def plot_sample_predictions(self, X_test, y_true, y_pred, num_samples=12):
        """
        Plot sample images with their predictions, highlighting misclassifications.
        
        Args:
            X_test: Test images
            y_true: True labels
            y_pred: Predicted labels
            num_samples: Number of samples to display
        """
        # Find misclassified examples
        misclassified = np.where(y_true != y_pred)[0]
        correct = np.where(y_true == y_pred)[0]
        
        # Select a mix of correct and incorrect predictions
        if len(misclassified) >= num_samples//2:
            selected_wrong = np.random.choice(misclassified, num_samples//2, replace=False)
            selected_correct = np.random.choice(correct, num_samples//2, replace=False)
            indices = np.concatenate([selected_wrong, selected_correct])
        else:
            indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(3, 4, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            axes[i].imshow(X_test[idx])
            
            true_label = self.classes[y_true[idx]]
            pred_label = self.classes[y_pred[idx]]
            
            if y_true[idx] == y_pred[idx]:
                color = 'green'
                title = f'✓ {true_label}'
            else:
                color = 'red'
                title = f'✗ True: {true_label}\nPred: {pred_label}'
            
            axes[i].set_title(title, color=color, fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('plots/sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_roc_auc(self, y_true, y_pred_proba):
        """
        Calculate ROC-AUC for multiclass classification (one-vs-rest).
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            tuple: (auc_scores, macro_auc) per-class and macro AUC scores
        """
        try:
            # Convert to one-hot encoding for multiclass ROC-AUC
            y_true_onehot = to_categorical(y_true, num_classes=10)
            
            # Calculate AUC for each class (one-vs-rest)
            auc_scores = []
            for i in range(10):
                auc_score = roc_auc_score(y_true_onehot[:, i], y_pred_proba[:, i])
                auc_scores.append(auc_score)
            
            macro_auc = np.mean(auc_scores)
            
            print(f"\nROC-AUC Scores (One-vs-Rest):")
            for i, (class_name, auc_score) in enumerate(zip(self.classes, auc_scores)):
                print(f"{class_name:12}: {auc_score:.4f}")
            print(f"{'Macro AUC':12}: {macro_auc:.4f}")
            
            # Save AUC scores
            auc_df = pd.DataFrame({
                'Class': self.classes,
                'AUC_Score': auc_scores
            })
            auc_df = pd.concat([auc_df, pd.DataFrame({
                'Class': ['Macro_AUC'], 
                'AUC_Score': [macro_auc]
            })], ignore_index=True)
            auc_df.to_csv('reports/roc_auc_scores.csv', index=False)
            
            return auc_scores, macro_auc
            
        except Exception as e:
            print(f"Could not calculate ROC-AUC: {e}")
            return None, None