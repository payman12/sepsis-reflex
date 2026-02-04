"""
Simple Transformer Model for Sepsis Prediction
================================================

A standalone implementation of the TimeSeriesTransformer for use when
the MIMIC-sepsis/src module is not available.

This is a simplified version that maintains compatibility with the
training bridge.
"""

import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class SimpleTransformer(nn.Module):
    """
    Simple Transformer for time series classification.
    
    Compatible with the TimeSeriesTransformer interface from MIMIC-sepsis.
    
    Architecture:
    1. Input projection to hidden dimension
    2. Positional encoding
    3. Transformer encoder layers
    4. Output projection with sigmoid for binary classification
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        task_type: str = 'classification'
    ):
        super().__init__()
        self.task_type = task_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.input_dim = input_dim
        self.initialized = False
        
        if input_dim is not None:
            self._initialize_model(input_dim)
            self.initialized = True
    
    def _initialize_model(self, input_dim: int):
        """Initialize model architecture"""
        self.input_dim = input_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
        if self.task_type == 'classification':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor (batch_size,)
        """
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Take last timestep
        x = x[:, -1, :]
        
        # Output projection
        x = self.output_layer(x)
        x = self.output_activation(x)
        
        return x.squeeze()
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        batch_size: int = 32,
        epochs: int = 10,
        learning_rate: float = 0.001,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """
        Train the model.
        
        Args:
            X: Features (n_samples, seq_len, n_features)
            y: Labels (n_samples,)
            batch_size: Batch size
            epochs: Number of epochs
            learning_rate: Learning rate
            validation_data: Optional (X_val, y_val) tuple
        """
        # Initialize if needed
        if not self.initialized:
            self.input_dim = X.shape[2]
            self._initialize_model(self.input_dim)
            self.initialized = True
        
        self.to(self.device)
        
        # Create dataloaders
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        val_dataloader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss() if self.task_type == 'classification' else nn.MSELoss()
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            # Validation
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader, criterion)
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    def _validate(self, val_dataloader, criterion) -> float:
        """Run validation and return loss"""
        self.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
        
        self.train()
        return total_loss / len(val_dataloader)
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features (n_samples, seq_len, n_features)
            batch_size: Batch size
            
        Returns:
            Predictions (n_samples,)
        """
        self.to(self.device)
        self.eval()
        
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        with torch.no_grad():
            for batch_X, in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self(batch_X)
                
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                
                batch_preds = outputs.cpu().numpy()
                if batch_preds.ndim == 0:
                    batch_preds = np.array([batch_preds.item()])
                
                predictions.append(batch_preds)
        
        if predictions:
            return np.concatenate(predictions)
        return np.array([])
