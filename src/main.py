"""
Main execution script for CIFAR-10 CNN Classifier
This script orchestrates the complete training and evaluation pipeline.
"""

import os
import argparse
import sys
from train import CIFAR10Classifier


def setup_directories():
    """Create necessary directories for outputs."""
    directories = ['models', 'plots', 'reports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created/verified directory: {directory}/")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN Classifier')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Maximum number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots (faster execution)')
    parser.add_argument('--model-only', action='store_true',
                       help='Only train model, skip evaluation and plots')
    
    return parser.parse_args()


def print_header():
    """Print a nice header for the application."""
    print("=" * 60)
    print("CIFAR-10 CNN CLASSIFIER")
    print("=" * 60)
    print("10-class image classification using Convolutional Neural Networks")
    print("Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")
    print("=" * 60)


def print_separator(text):
    """Print a section separator."""
    print(f"\n{'='*20} {text} {'='*20}")


def run_training_pipeline(args):
    """
    Run the complete training and evaluation pipeline.
    
    Args:
        args: Parsed command line arguments
    """
    # Initialize classifier
    classifier = CIFAR10Classifier()
    
    # Load and preprocess data
    print_separator("DATA LOADING")
    X_train, y_train, X_test, y_test = classifier.load_and_preprocess_data()
    
    # Train model
    print_separator("MODEL TRAINING")
    classifier.train_model(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    
    # If model-only mode, return early
    if args.model_only:
        print("\nModel training completed! (model-only mode)")
        print("Model saved to: models/")
        return
    
    # Evaluate model
    print_separator("MODEL EVALUATION")
    y_pred, y_pred_proba = classifier.evaluate_model(X_test, y_test)
    
    # Generate visualizations (unless skipped)
    if not args.skip_plots:
        print_separator("GENERATING VISUALIZATIONS")
        
        print("Generating learning curves...")
        classifier.plot_learning_curves()
        
        print("Generating confusion matrix...")
        classifier.plot_confusion_matrix(y_test, y_pred)
        
        print("Generating per-class metrics plot...")
        classifier.plot_per_class_metrics(y_test, y_pred)
        
        print("Generating sample predictions...")
        classifier.plot_sample_predictions(X_test, y_test, y_pred)
    
    # Calculate additional metrics
    print_separator("ADDITIONAL METRICS")
    classifier.calculate_roc_auc(y_test, y_pred_proba)
    
    return classifier


def print_summary(args):
    """Print a summary of outputs and next steps."""
    print_separator("EXECUTION SUMMARY")
    print("Training and evaluation completed successfully!")
    print("\nGenerated outputs:")
    print("   └── models/")
    print("       ├── best_model.h5      (best weights during training)")
    print("       └── final_model.h5     (final model)")
    print("   └── reports/")
    print("       ├── classification_report.txt")
    print("       ├── per_class_metrics.csv")
    print("       └── roc_auc_scores.csv")
    
    if not args.skip_plots:
        print("   └── plots/")
        print("       ├── learning_curves.png")
        print("       ├── confusion_matrix.png")
        print("       ├── per_class_metrics.png")
        print("       └── sample_predictions.png")
    
    print(f"\nTraining configuration:")
    print(f"   • Max epochs: {args.epochs}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Plots generated: {not args.skip_plots}")
    
    print("\nNext steps:")
    print("   • Load model: tf.keras.models.load_model('models/best_model.h5')")
    print("   • View reports: Check files in reports/ directory")
    print("   • Analyze plots: Check files in plots/ directory")


def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print header
        print_header()
        
        # Setup directories
        print_separator("SETUP")
        setup_directories()
        
        # Run training pipeline
        classifier = run_training_pipeline(args)
        
        # Print summary
        print_summary(args)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Partially trained model may be saved in models/ directory")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check your environment and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()