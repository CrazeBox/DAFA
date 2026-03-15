"""LSTM model for Shakespeare character-level language modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LSTMModel(nn.Module):
    """LSTM model for sequence modeling."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 8,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_size: Hidden size of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            hidden: Initial hidden state
        
        Returns:
            Tuple of (output, hidden_state)
        """
        batch_size = x.size(0)
        
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        lstm_out = self.dropout(lstm_out)
        
        output = self.fc(lstm_out[:, -1, :])
        
        return output, hidden
    
    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
        
        Returns:
            Tuple of (hidden_state, cell_state)
        """
        weight = next(self.parameters())
        
        hidden = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
        cell = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
        
        return (hidden, cell)
    
    def get_features(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get features before the final classification layer."""
        batch_size = x.size(0)
        
        embedded = self.embedding(x)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        lstm_out, _ = self.lstm(embedded, hidden)
        
        return lstm_out[:, -1, :]


class ShakespeareLSTM(nn.Module):
    """LSTM model optimized for Shakespeare dataset."""
    
    def __init__(
        self,
        vocab_size: int = 80,
        embedding_dim: int = 8,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """
        Initialize Shakespeare LSTM.
        
        Args:
            vocab_size: Size of vocabulary (default: 80 for Shakespeare)
            embedding_dim: Dimension of character embeddings
            hidden_size: Hidden size of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm_layers = nn.ModuleList([
            nn.LSTMCell(
                embedding_dim if i == 0 else hidden_size,
                hidden_size,
            )
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights."""
        for lstm in self.lstm_layers:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            hidden: Initial hidden state tuple (h, c) for each layer
        
        Returns:
            Tuple of (output, hidden_state)
        """
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        h, c = hidden
        
        embedded = self.embedding(x)
        
        outputs = []
        for t in range(seq_length):
            x_t = embedded[:, t, :]
            
            for i, lstm in enumerate(self.lstm_layers):
                h[i], c[i] = lstm(x_t, (h[i], c[i]))
                x_t = h[i]
            
            outputs.append(h[-1])
        
        output = torch.stack(outputs, dim=1)
        output = self.dropout(output[:, -1, :])
        output = self.fc(output)
        
        return output, (h, c)
    
    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
        
        Returns:
            Tuple of (hidden_states, cell_states) for all layers
        """
        weight = next(self.parameters())
        
        h = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
        c = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
        
        return (h, c)
    
    def get_features(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get features before the final classification layer."""
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        h, c = hidden
        
        embedded = self.embedding(x)
        
        for t in range(x.size(1)):
            x_t = embedded[:, t, :]
            
            for i, lstm in enumerate(self.lstm_layers):
                h[i], c[i] = lstm(x_t, (h[i], c[i]))
                x_t = h[i]
        
        return h[-1]


class StackedLSTM(nn.Module):
    """Stacked LSTM with residual connections."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 8,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """
        Initialize Stacked LSTM.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_size: Hidden size of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstms = nn.ModuleList()
        self.lstms.append(nn.LSTM(embedding_dim, hidden_size, batch_first=True))
        for _ in range(num_layers - 1):
            self.lstms.append(nn.LSTM(hidden_size, hidden_size, batch_first=True))
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights."""
        for lstm in self.lstms:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass."""
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        hiddens = []
        output = embedded
        
        for i, lstm in enumerate(self.lstms):
            output, hidden_i = lstm(output)
            output = self.dropout(output)
            hiddens.append(hidden_i)
        
        output = self.fc(output[:, -1, :])
        
        return output, hiddens[-1] if hiddens else None
