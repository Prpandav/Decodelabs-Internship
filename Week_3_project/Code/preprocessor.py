"""
Text Preprocessor and Model Utilities
Shared between Jupyter Notebook training and Streamlit Web Application.
"""

import re
import json
import os
import torch
import torch.nn as nn
import numpy as np

def clean_text(text: str) -> str:
    """
    Standard text normalization:
    - Lowercase
    - Replace non-alphanumeric characters with space
    - Collapse extra whitespaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_input_text(headline: str, content: str) -> str:
    """
    Combines headline and content for optimal contextual signal.
    """
    h = headline.strip() if headline else ""
    c = content.strip() if content else ""
    if h and c:
        return f"{h} - {c}"
    return h or c

class TextCNN(nn.Module):
    """
    Multi-Filter 1D Convolutional Neural Network for Text Classification.
    """
    def __init__(self, vocab_size, embed_dim=128, num_classes=10, num_filters=64, filter_sizes=[2, 3, 4]):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
    def forward(self, x):
        # x: [batch, max_len]
        emb = self.embedding(x).transpose(1, 2) # [batch, embed_dim, max_len]
        conv_outs = [torch.relu(conv(emb)) for conv in self.convs]
        pooled = [torch.max(co, dim=2)[0] for co in conv_outs] # Global max pool
        cat = self.dropout(torch.cat(pooled, dim=1))
        out = self.fc(cat)
        return out

def text_to_sequence(text: str, vocab: dict, max_len: int = 100):
    """
    Tokenizes and converts clean text into a fixed-length numerical tensor.
    """
    tokens = clean_text(text).split()[:max_len]
    seq = [vocab.get(w, vocab.get("<UNK>", 1)) for w in tokens]
    if len(seq) < max_len:
        seq += [vocab.get("<PAD>", 0)] * (max_len - len(seq))
    return torch.tensor([seq], dtype=torch.long)

