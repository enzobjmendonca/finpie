from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import bitsandbytes as bnb
from datasets import Dataset

from ..data.timeseries import TimeSeries

class MarketTokenizer:
    """
    Class for converting market data into tokenized format suitable for LLM processing.
    """
    
    def __init__(self, 
                 price_precision: int = 4,
                 volume_precision: int = 2,
                 time_format: str = "%Y-%m-%d %H:%M:%S"):
        """
        Initialize MarketTokenizer.
        
        Args:
            price_precision: Number of decimal places for price values
            volume_precision: Number of decimal places for volume values
            time_format: Format string for datetime conversion
        """
        self.price_precision = price_precision
        self.volume_precision = volume_precision
        self.time_format = time_format
        
    def _format_value(self, value: float, precision: int) -> str:
        """Format numerical value with specified precision."""
        return f"{value:.{precision}f}"
    
    def _create_market_token(self, 
                           timestamp: datetime,
                           price: float,
                           volume: float,
                           additional_features: Optional[Dict] = None) -> str:
        """
        Create a single market data token.
        
        Args:
            timestamp: Time of the data point
            price: Price value
            volume: Volume value
            additional_features: Optional dictionary of additional features
            
        Returns:
            Formatted token string
        """
        token = f"T:{timestamp.strftime(self.time_format)} P:{self._format_value(price, self.price_precision)} V:{self._format_value(volume, self.volume_precision)}"
        
        if additional_features:
            for key, value in additional_features.items():
                if isinstance(value, (int, float)):
                    token += f" {key}:{self._format_value(value, self.price_precision)}"
                else:
                    token += f" {key}:{value}"
                    
        return token
    
    def tokenize_sequence(self, 
                         data: pd.DataFrame,
                         sequence_length: int,
                         target_length: int,
                         step: int = 1) -> List[Tuple[str, str]]:
        """
        Convert market data into sequences of tokens for LLM processing.
        
        Args:
            data: DataFrame with market data (must have datetime index)
            sequence_length: Number of past points to include in input sequence
            target_length: Number of future points to predict
            step: Step size for sliding window
            
        Returns:
            List of tuples containing (input_sequence, target_sequence)
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have datetime index")
            
        sequences = []
        total_length = len(data)
        
        for i in range(0, total_length - sequence_length - target_length + 1, step):
            # Input sequence
            input_data = data.iloc[i:i + sequence_length]
            input_tokens = []
            
            for idx, row in input_data.iterrows():
                token = self._create_market_token(
                    timestamp=idx,
                    price=row['close'] if 'close' in row else row.iloc[0],
                    volume=row['volume'] if 'volume' in row else 0,
                    additional_features={k: v for k, v in row.items() if k not in ['close', 'volume']}
                )
                input_tokens.append(token)
            
            # Target sequence
            target_data = data.iloc[i + sequence_length:i + sequence_length + target_length]
            target_tokens = []
            
            for idx, row in target_data.iterrows():
                token = self._create_market_token(
                    timestamp=idx,
                    price=row['close'] if 'close' in row else row.iloc[0],
                    volume=row['volume'] if 'volume' in row else 0,
                    additional_features={k: v for k, v in row.items() if k not in ['close', 'volume']}
                )
                target_tokens.append(token)
            
            sequences.append((' '.join(input_tokens), ' '.join(target_tokens)))
            
        return sequences

class LLMForecaster:
    """
    Class for forecasting market movements using LLM-based approach.
    """
    
    def __init__(self, 
                 model_name: str,
                 tokenizer: Optional[MarketTokenizer] = None,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 quantization: str = "4bit"):
        """
        Initialize LLMForecaster.
        
        Args:
            model_name: Name of the LLM model to use
            tokenizer: Optional MarketTokenizer instance
            lora_r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            quantization: Quantization type ("4bit" or "8bit")
        """
        self.model_name = model_name
        self.tokenizer = tokenizer or MarketTokenizer()
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.quantization = quantization
        
        # Initialize model and tokenizer
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model with LoRA and quantization."""
        # Load base model with quantization
        if self.quantization == "4bit":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:  # 8bit
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=True,
                device_map="auto"
            )
            
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj"],  # Target attention modules
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Get PEFT model
        self.model = get_peft_model(self.model, lora_config)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def _prepare_dataset(self, training_data: List[Tuple[str, str]]) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            training_data: List of (input_sequence, target_sequence) pairs
            
        Returns:
            HuggingFace Dataset
        """
        # Combine input and target sequences
        texts = [f"{input_seq} {target_seq}" for input_seq, target_seq in training_data]
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,  # Adjust based on your needs
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"]
        })
        
        return dataset
    
    def train(self, 
             training_data: List[Tuple[str, str]],
             epochs: int = 10,
             batch_size: int = 32,
             learning_rate: float = 2e-4,
             warmup_steps: int = 100,
             output_dir: str = "market_forecaster_model") -> Dict:
        """
        Train the LLM model on market data sequences using LoRA and quantization.
        
        Args:
            training_data: List of (input_sequence, target_sequence) pairs
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            warmup_steps: Number of warmup steps
            output_dir: Directory to save the model
            
        Returns:
            Dictionary containing training metrics
        """
        # Prepare dataset
        dataset = self._prepare_dataset(training_data)
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            fp16=True,
            optim="paged_adamw_8bit"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        
        # Return training metrics
        return {
            "train_loss": trainer.state.log_history[-1]["loss"],
            "epochs_trained": epochs,
            "model_saved": True,
            "output_dir": output_dir
        }
    
    def predict(self, 
               input_sequence: str,
               max_tokens: int = 100) -> str:
        """
        Generate market predictions using the trained LLM.
        
        Args:
            input_sequence: Tokenized input sequence
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Predicted market sequence
        """
        # Tokenize input
        inputs = self.tokenizer(
            input_sequence,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        
        # Generate predictions
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return prediction
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def evaluate(self,
                test_data: List[Tuple[str, str]]) -> Dict:
        """
        Evaluate the model's performance on test data.
        
        Args:
            test_data: List of (input_sequence, target_sequence) pairs
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Prepare test dataset
        test_dataset = self._prepare_dataset(test_data)
        
        # Configure evaluation arguments
        eval_args = TrainingArguments(
            output_dir="eval_results",
            per_device_eval_batch_size=8,
            evaluation_strategy="no"
        )
        
        # Initialize trainer for evaluation
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=test_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Run evaluation
        eval_results = trainer.evaluate()
        
        return {
            "eval_loss": eval_results["eval_loss"],
            "perplexity": np.exp(eval_results["eval_loss"])
        } 