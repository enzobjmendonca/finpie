"""
High-level forecasting interface for LLM-based financial predictions.

This module provides the main LLMForecaster class that orchestrates the entire
forecasting pipeline from data preprocessing to model training and prediction.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List

from .tokenizer import MarketTokenizer
from .dataset import ReturnTokenDataset
from .transformer import MarketTransformer, DEVICE


class LLMForecaster:
    """
    High-level interface for training and using the transformer model for market forecasting.
    
    This class provides a convenient interface for:
    - Building and training the transformer model
    - Generating market forecasts
    - Handling data preprocessing and tokenization
    
    Attributes:
        seq_len (int): Length of input sequences
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        vocab_size (int): Size of the token vocabulary
        emb_dim (int): Dimension of token embeddings
        lr (float): Learning rate
        
    Example:
        >>> forecaster = LLMForecaster(seq_len=60, batch_size=64)
        >>> forecaster.fit(returns)
        >>> predictions = forecaster.predict(prompt)
    """

    def __init__(self, seq_len=60, batch_size=256, vocab_size=64, emb_dim=64, lr=3e-4, device=None):
        """
        Initialize the LLM forecaster.
        
        Args:
            seq_len: Length of input sequences
            batch_size: Batch size for training
            vocab_size: Size of the token vocabulary
            emb_dim: Dimension of token embeddings
            lr: Learning rate
            device: Device to use (if None, will auto-detect)
        """
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.lr = lr
        self.device = device if device is not None else DEVICE
        
        self.tokenizer = MarketTokenizer()
        self.model = None
        self.optimizer = None
        self.dataset = None
        self.dataloader = None
        
        # Training metrics tracking
        self.loss_history = []
        self.batch_losses = []
        self.learning_rates = []
        self.gradient_norms = []
        self.training_metrics = {}
        
        # Print device information
        self._print_device_info()

    def _print_device_info(self):
        """Print information about the device being used."""
        print(f"LLMForecaster initialized with device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(self.device)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("Running on CPU")

    def build_dataset(self, returns: pd.Series):
        """
        Build the dataset for training.
        
        Args:
            returns: Series of market returns
            
        Example:
            >>> forecaster.build_dataset(returns)
        """
        tokens, self.bins, self.original_mean = self.tokenizer.series_to_tokens(
            returns, self.vocab_size, method='equal_freq', zero_centered=True
        )
        self.dataset = ReturnTokenDataset(tokens, self.seq_len)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=True,  # Faster CPU-GPU transfer
            num_workers=2     # Parallel data loading
        )

    def build_model(self):
        """
        Build the transformer model.
        
        Example:
            >>> forecaster.build_model()
        """
        self.model = MarketTransformer(self.vocab_size, self.emb_dim)
        # Ensure model is on the correct device
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, epochs=5, track_metrics=True):
        """
        Train the model on the prepared dataset with detailed metrics tracking.
        
        Args:
            epochs: Number of training epochs
            track_metrics: Whether to track detailed training metrics
            
        Example:
            >>> forecaster.train()
        """
        if self.model is None:
            self.build_model()
            
        criterion = nn.CrossEntropyLoss()
        
        # Initialize tracking
        if track_metrics:
            self.loss_history = []
            self.batch_losses = []
            self.gradient_norms = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            epoch_batch_losses = []
            epoch_grad_norms = []

            for batch_idx, (batch_x, batch_y) in enumerate(self.dataloader):
                # Move batch data to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output.view(-1, self.vocab_size), batch_y.view(-1))
                loss.backward()
                
                # Track gradient norm
                if track_metrics:
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    epoch_grad_norms.append(total_norm)
                
                self.optimizer.step()
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                if track_metrics:
                    epoch_batch_losses.append(batch_loss)
                
            avg_loss = epoch_loss / len(self.dataloader)
            
            # Store metrics
            if track_metrics:
                self.loss_history.append(avg_loss)
                self.batch_losses.extend(epoch_batch_losses)
                self.gradient_norms.extend(epoch_grad_norms)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Real-time loss analysis
            if track_metrics and len(self.loss_history) > 1:
                self._analyze_epoch_progress(epoch + 1, avg_loss)

    def fit(self, prompt, steps=10, temperature=1.0, top_k=None):
        """
        Generate a forecast sequence from a prompt with improved sampling.
        
        Args:
            prompt: Initial sequence of returns
            steps: Number of steps to forecast
            temperature: Sampling temperature (1.0 = no change, <1.0 = more conservative, >1.0 = more diverse)
            top_k: If specified, only sample from top-k tokens
            
        Returns:
            Series of forecasted returns
            
        Example:
            >>> forecast = forecaster.fit(prompt, steps=10, temperature=1.2)
            >>> # forecast contains the predicted returns with more diversity
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        self.model.eval()
        
        # Use EXISTING bins from training - never recreate bins during prediction!
        if not hasattr(self, 'bins') or self.bins is None:
            raise ValueError("Model bins not available. Train the model first or load a trained model.")
        
        # Center the prompt data using the training mean
        if hasattr(self, 'original_mean'):
            centered_prompt = prompt - self.original_mean
        else:
            centered_prompt = prompt
            
        # Tokenize using EXISTING bins from training
        tokens = np.digitize(centered_prompt, self.bins[:-1])
        tokens = torch.tensor(tokens[-self.seq_len:], dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(steps):
                output = self.model(tokens)
                logits = output[0, -1]
                
                # Apply temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    top_logits, top_indices = torch.topk(logits, top_k)
                    # Set all other logits to -inf
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(0, top_indices, top_logits)
                    logits = logits_filtered
                
                # Sample from the distribution instead of taking argmax
                probabilities = torch.softmax(logits, dim=0)
                next_token = torch.multinomial(probabilities, 1)
                
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                tokens = tokens[:, -self.seq_len:]
                
        forecast_tokens = tokens[0, -steps:].cpu().numpy()
        forecast_values = self.tokenizer.tokens_to_values(forecast_tokens, self.bins, self.original_mean)
        
        return pd.Series(forecast_values, index=pd.date_range(start=prompt.index[-1], periods=steps+1)[1:])
    
    def _analyze_epoch_progress(self, epoch: int, current_loss: float):
        """Analyze training progress in real-time."""
        if len(self.loss_history) < 2:
            return
            
        # Check for loss explosion
        if current_loss > self.loss_history[0] * 2:
            print(f"   WARNING: Loss explosion detected at epoch {epoch}")
        
        # Check for plateau
        if len(self.loss_history) >= 3:
            recent_losses = self.loss_history[-3:]
            if max(recent_losses) - min(recent_losses) < 0.001:
                print(f"   INFO: Loss plateau detected (variance < 0.001)")
        
        # Check improvement rate
        improvement = (self.loss_history[0] - current_loss) / self.loss_history[0] * 100
        print(f"   Progress: {improvement:.1f}% improvement from initial loss")
    
    def analyze_training_metrics(self):
        """
        Comprehensive analysis of training metrics.
        
        Returns:
            Dictionary containing analysis results
        """
        if not self.loss_history:
            print("ERROR: No training metrics available. Train the model first.")
            return {}
        
        print("\nCOMPREHENSIVE TRAINING ANALYSIS")
        print("=" * 50)
        
        analysis = {}
        
        # 1. Loss Analysis
        losses = np.array(self.loss_history)
        analysis['loss'] = {
            'initial': float(losses[0]),
            'final': float(losses[-1]),
            'min': float(losses.min()),
            'max': float(losses.max()),
            'improvement_pct': float((losses[0] - losses[-1]) / losses[0] * 100),
            'stability_cv': float(np.std(losses) / np.mean(losses))
        }
        
        print(f"Loss Metrics:")
        print(f"   Initial -> Final: {losses[0]:.4f} -> {losses[-1]:.4f}")
        print(f"   Improvement: {analysis['loss']['improvement_pct']:.1f}%")
        print(f"   Stability (CV): {analysis['loss']['stability_cv']:.4f}")
        
        # 2. Gradient Analysis
        if self.gradient_norms:
            grads = np.array(self.gradient_norms)
            analysis['gradients'] = {
                'mean': float(grads.mean()),
                'std': float(grads.std()),
                'max': float(grads.max()),
                'min': float(grads.min())
            }
            
            print(f"\nGradient Metrics:")
            print(f"   Mean norm: {grads.mean():.6f}")
            print(f"   Max norm: {grads.max():.6f}")
            if grads.max() > 10:
                print("   WARNING: High gradient norms detected")
            elif grads.mean() < 1e-6:
                print("   WARNING: Very small gradients detected")
        
        # 3. Theoretical Comparison
        random_loss = np.log(self.vocab_size)
        good_target = random_loss * 0.3
        
        analysis['performance'] = {
            'random_baseline': float(random_loss),
            'good_target': float(good_target),
            'beats_random': bool(losses[-1] < random_loss),
            'reaches_good': bool(losses[-1] < good_target)
        }
        
        print(f"\nPerformance Assessment:")
        print(f"   Random baseline: {random_loss:.4f}")
        print(f"   Good target: {good_target:.4f}")
        print(f"   Your model: {losses[-1]:.4f}")
        
        if analysis['performance']['reaches_good']:
            print("   EXCELLENT: Reached good performance target!")
        elif analysis['performance']['beats_random']:
            print("   GOOD: Better than random baseline")
        else:
            print("   NEEDS WORK: Not yet better than random")
        
        # 4. Training Recommendations
        recommendations = self._generate_recommendations(analysis)
        if recommendations:
            print(f"\nRecommendations:")
            for rec in recommendations:
                print(f"   - {rec}")
        
        return analysis
    
    def _generate_recommendations(self, analysis: dict) -> List[str]:
        """Generate training recommendations based on analysis."""
        recommendations = []
        
        # Loss-based recommendations
        if analysis['loss']['improvement_pct'] < 10:
            recommendations.append("Consider training for more epochs or increasing learning rate")
        
        if analysis['loss']['stability_cv'] > 0.3:
            recommendations.append("High loss variance - consider reducing learning rate")
        
        if not analysis['performance']['beats_random']:
            recommendations.append("Model not learning - check data quality and model architecture")
        
        # Gradient-based recommendations
        if 'gradients' in analysis:
            if analysis['gradients']['max'] > 10:
                recommendations.append("Large gradients - consider gradient clipping or lower learning rate")
            elif analysis['gradients']['mean'] < 1e-6:
                recommendations.append("Small gradients - consider higher learning rate or check for vanishing gradients")
        
        return recommendations
    
    def plot_training_metrics(self, save_path: str = None):
        """
        Plot comprehensive training metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.loss_history:
            print("ERROR: No training metrics available. Train the model first.")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Training Metrics Analysis', fontsize=16, fontweight='bold')
            
            # 1. Loss curve
            ax = axes[0, 0]
            epochs = range(1, len(self.loss_history) + 1)
            ax.plot(epochs, self.loss_history, 'b-', linewidth=2, marker='o')
            ax.set_title('Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            
            # Add random baseline
            random_loss = np.log(self.vocab_size)
            ax.axhline(y=random_loss, color='r', linestyle='--', alpha=0.7, label='Random baseline')
            ax.legend()
            
            # 2. Gradient norms
            if self.gradient_norms:
                ax = axes[0, 1]
                ax.plot(self.gradient_norms, 'g-', alpha=0.7)
                ax.set_title('Gradient Norms')
                ax.set_xlabel('Batch')
                ax.set_ylabel('Gradient Norm')
                ax.grid(True, alpha=0.3)
            
            # 3. Loss distribution
            if self.batch_losses:
                ax = axes[1, 0]
                ax.hist(self.batch_losses, bins=30, alpha=0.7, color='orange')
                ax.set_title('Batch Loss Distribution')
                ax.set_xlabel('Loss')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            
            # 4. Learning progress
            ax = axes[1, 1]
            if len(self.loss_history) > 1:
                improvements = [(self.loss_history[0] - loss) / self.loss_history[0] * 100 
                              for loss in self.loss_history]
                ax.plot(epochs, improvements, 'm-', linewidth=2, marker='s')
                ax.set_title('Learning Progress')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Improvement (%)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Training metrics plot saved to: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("ERROR: Matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"ERROR: Error creating plot: {e}")
    
    def get_training_summary(self) -> dict:
        """
        Get a concise summary of training results.
        
        Returns:
            Dictionary with key training metrics
        """
        if not self.loss_history:
            return {"error": "No training data available"}
        
        losses = np.array(self.loss_history)
        random_baseline = np.log(self.vocab_size)
        
        summary = {
            "epochs_trained": len(self.loss_history),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "best_loss": float(losses.min()),
            "improvement_percent": float((losses[0] - losses[-1]) / losses[0] * 100),
            "beats_random": bool(losses[-1] < random_baseline),
            "convergence_status": self._assess_convergence(),
            "training_stability": "stable" if np.std(losses) / np.mean(losses) < 0.1 else "unstable"
        }
        
        return summary
    
    def _assess_convergence(self) -> str:
        """Assess convergence status of training."""
        if len(self.loss_history) < 3:
            return "insufficient_data"
        
        recent_losses = self.loss_history[-3:]
        slope = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
        
        if abs(slope) < 0.001:
            return "converged"
        elif slope < -0.001:
            return "improving"
        else:
            return "diverging"
    
    def save_model(self, filepath: str, include_training_state: bool = True):
        """
        Save the trained model and all necessary components.
        
        Args:
            filepath: Path to save the model (without extension)
            include_training_state: Whether to save optimizer state and training metrics
            
        Example:
            >>> forecaster.save_model("my_financial_model")
            >>> # Creates: my_financial_model.pth
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        # Prepare save data
        save_data = {
            # Model architecture parameters
            'model_config': {
                'vocab_size': self.vocab_size,
                'emb_dim': self.emb_dim,
                'seq_len': self.seq_len,
                'batch_size': self.batch_size,
                'lr': self.lr
            },
            
            # Model weights
            'model_state_dict': self.model.state_dict(),
            
            # Tokenizer data (essential for predictions)
            'tokenizer_bins': self.bins if hasattr(self, 'bins') else None,
            'original_mean': self.original_mean if hasattr(self, 'original_mean') else 0.0,
            
            # Training metadata
            'training_metadata': {
                'loss_history': self.loss_history,
                'final_loss': self.loss_history[-1] if self.loss_history else None,
                'epochs_trained': len(self.loss_history),
                'convergence_status': self._assess_convergence() if self.loss_history else None
            }
        }
        
        # Optionally include training state
        if include_training_state and self.optimizer is not None:
            save_data['optimizer_state_dict'] = self.optimizer.state_dict()
            save_data['training_metrics'] = {
                'batch_losses': self.batch_losses,
                'gradient_norms': self.gradient_norms,
            }
        
        # Save to file
        full_path = f"{filepath}.pth"
        torch.save(save_data, full_path)
        
        print(f"Model saved successfully to: {full_path}")
        print(f"   Model config: vocab_size={self.vocab_size}, emb_dim={self.emb_dim}")
        print(f"   Training history: {len(self.loss_history)} epochs")
        if self.loss_history:
            print(f"   Final loss: {self.loss_history[-1]:.4f}")
        
        return full_path
    
    def load_model(self, filepath: str, device: str = None):
        """
        Load a previously saved model.
        
        Args:
            filepath: Path to the saved model file
            device: Device to load model on (if None, uses current device)
            
        Example:
            >>> forecaster = LLMForecaster()
            >>> forecaster.load_model("my_financial_model.pth")
            >>> # Now ready for predictions!
        """
        # Add .pth extension if not present
        if not filepath.endswith('.pth'):
            filepath = f"{filepath}.pth"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load data
        print(f"Loading model from: {filepath}")
        
        if device is None:
            device = self.device
        
        # Load on specified device (set weights_only=False for compatibility)
        save_data = torch.load(filepath, map_location=device, weights_only=False)
        
        # Restore model configuration
        config = save_data['model_config']
        self.vocab_size = config['vocab_size']
        self.emb_dim = config['emb_dim']
        self.seq_len = config['seq_len']
        self.batch_size = config['batch_size']
        self.lr = config['lr']
        
        # Rebuild model with loaded config
        self.model = MarketTransformer(self.vocab_size, self.emb_dim)
        self.model.to(device)
        
        # Load model weights
        self.model.load_state_dict(save_data['model_state_dict'])
        
        # Restore tokenizer bins (crucial for predictions)
        if save_data['tokenizer_bins'] is not None:
            self.bins = save_data['tokenizer_bins']
            self.original_mean = save_data.get('original_mean', 0.0)
        else:
            print("WARNING: No tokenizer bins found. You'll need to rebuild dataset before predictions.")
        
        # Restore training history
        metadata = save_data['training_metadata']
        self.loss_history = metadata['loss_history']
        
        # Optionally restore optimizer state
        if 'optimizer_state_dict' in save_data:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.optimizer.load_state_dict(save_data['optimizer_state_dict'])
            print("Optimizer state restored (can resume training)")
        
        # Restore additional training metrics
        if 'training_metrics' in save_data:
            self.batch_losses = save_data['training_metrics']['batch_losses']
            self.gradient_norms = save_data['training_metrics']['gradient_norms']
        
        print(f"Model loaded successfully!")
        print(f"   Architecture: vocab_size={self.vocab_size}, emb_dim={self.emb_dim}")
        print(f"   Training history: {len(self.loss_history)} epochs")
        if self.loss_history:
            print(f"   Final training loss: {self.loss_history[-1]:.4f}")
            print(f"   Convergence status: {self._assess_convergence()}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        return self
    
    @classmethod
    def from_saved_model(cls, filepath: str, device: str = None):
        """
        Create a new LLMForecaster instance from a saved model.
        
        Args:
            filepath: Path to the saved model file
            device: Device to load model on
            
        Returns:
            New LLMForecaster instance with loaded model
            
        Example:
            >>> forecaster = LLMForecaster.from_saved_model("my_financial_model.pth")
            >>> forecast = forecaster.fit(recent_data, steps=10)
        """
        # Create new instance with minimal initialization
        instance = cls.__new__(cls)
        
        # Set device
        if device is None:
            device = DEVICE
        instance.device = device
        
        # Initialize tokenizer and empty containers
        instance.tokenizer = MarketTokenizer()
        instance.dataset = None
        instance.dataloader = None
        instance.training_metrics = {}
        
        # Load the model
        instance.load_model(filepath, device)
        
        return instance
    
    def export_model_info(self, filepath: str = None):
        """
        Export detailed model information to a text file.
        
        Args:
            filepath: Path for the info file (optional)
            
        Returns:
            String containing model information
        """
        if self.model is None:
            return "No model available"
        
        info_lines = []
        info_lines.append("FINANCIAL LLM MODEL INFORMATION")
        info_lines.append("=" * 50)
        
        # Model architecture
        info_lines.append(f"\nModel Architecture:")
        info_lines.append(f"   Vocabulary Size: {self.vocab_size}")
        info_lines.append(f"   Embedding Dimension: {self.emb_dim}")
        info_lines.append(f"   Sequence Length: {self.seq_len}")
        info_lines.append(f"   Batch Size: {self.batch_size}")
        info_lines.append(f"   Learning Rate: {self.lr}")
        
        # Model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        info_lines.append(f"\nModel Parameters:")
        info_lines.append(f"   Total Parameters: {total_params:,}")
        info_lines.append(f"   Trainable Parameters: {trainable_params:,}")
        
        # Training information
        if self.loss_history:
            info_lines.append(f"\nTraining Information:")
            info_lines.append(f"   Epochs Trained: {len(self.loss_history)}")
            info_lines.append(f"   Initial Loss: {self.loss_history[0]:.4f}")
            info_lines.append(f"   Final Loss: {self.loss_history[-1]:.4f}")
            info_lines.append(f"   Best Loss: {min(self.loss_history):.4f}")
            improvement = (self.loss_history[0] - self.loss_history[-1]) / self.loss_history[0] * 100
            info_lines.append(f"   Improvement: {improvement:.1f}%")
            info_lines.append(f"   Convergence Status: {self._assess_convergence()}")
            
            # Performance vs baseline
            random_baseline = np.log(self.vocab_size)
            info_lines.append(f"\nPerformance Assessment:")
            info_lines.append(f"   Random Baseline: {random_baseline:.4f}")
            improvement_over_random = (random_baseline - self.loss_history[-1]) / random_baseline * 100
            info_lines.append(f"   Improvement over Random: {improvement_over_random:.1f}%")
            
            if self.loss_history[-1] < random_baseline * 0.3:
                status = "EXCELLENT"
            elif self.loss_history[-1] < random_baseline * 0.7:
                status = "GOOD"
            elif self.loss_history[-1] < random_baseline:
                status = "LEARNING"
            else:
                status = "NEEDS WORK"
            info_lines.append(f"   Status: {status}")
        
        # Device information
        info_lines.append(f"\nHardware Information:")
        info_lines.append(f"   Device: {self.device}")
        if torch.cuda.is_available():
            info_lines.append(f"   GPU: {torch.cuda.get_device_name(self.device)}")
            info_lines.append(f"   CUDA Version: {torch.version.cuda}")
        
        info_text = "\n".join(info_lines)
        
        # Save to file if requested
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(info_text)
            print(f"Model information saved to: {filepath}")
        
        return info_text
