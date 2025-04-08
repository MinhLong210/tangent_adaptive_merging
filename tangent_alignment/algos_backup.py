import itertools
import logging
import os
from pathlib import Path
from typing import List

import hydra
import lightning as L
import lightning.pytorch as pl
import requests
import torch
import torch.nn.functional as F
import torchmetrics
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from peta.metrics.accuracy import Accuracy
# from peta.models.clip import (
#     CLIP_MODELS,
#     freeze_unless_image_model,
#     get_lora_vision_model,
#     load_clip_model,
# )
from peta.models.LinearizedModel import LinearizedModelWraper
from peta.optim import CosineAnnealingWithWarmup
from peta.utils.logging import TitledLog, setup_colorlogging

def closed_form_linear(model, inputs, labels, low_rank=4):
    """
    Compute a closed-form-like update for a PyTorch model.
    
    Args:
        model: A linearized LinearizedModel(Huggingface) pretrained model that takes inputs and returns predictions.
        inputs: Input data (torch.Tensor, shape (n, d) where n is samples, d is features).
        labels: True labels (torch.Tensor, shape (n, K) for K classes or (n,) for regression).
    
    """
    
    labels = labels.squeeze()  # Shape: (9,)
    labels = F.one_hot(labels, num_classes=10)
    # Number of samples and output dimensions
    n, K = labels.shape
    
    # Forward pass: compute predictions
    model.eval()  # Set to evaluation mode (no dropout, etc.)
    predictions = model(**inputs)  # Shape: (n, K) or (n,)
    
    # Get model parameters and flatten them
    params = [p for p in model.parameters() if p.requires_grad] # only LoRA params are trainable
    total_num_params = sum(p.numel() for p in params)  # Total number of parameters
    param_shapes = [p.shape for p in params]  # To restore shapes later
    
    # Compute residual: labels - predictions
    b_vector = labels - predictions.logits  # Shape: (n, K)
    b_vector = b_vector.view(-1)  # Shape: (n * K,)
    
    # Compute Jacobian of predictions wrt parameters
    A = []
    for i in range(n * K):
        print(i)
        # Zero gradients
        model.zero_grad()
        # Compute gradient of the i-th output wrt all parameters
        predictions_flat = predictions.logits.view(-1)  # Shape: (n * K,)
        predictions_flat[i].backward(retain_graph=True)
        grad_flat = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                               for p in params])  # Shape: (p,)
        A.append(grad_flat)
    A = torch.stack(A)  # Shape: (n * K, p)

    # Compute A.T @ b
    At_b = torch.matmul(A.T, b_vector) # Shape: (p,)
    # global_Atb += At_b
    
    # Compute SVD of J
    U, S, Vh = torch.svd(A)  # U: (n*K, p), S: (p,), Vh: (p, n*K)
    # Compute eigen decomposition with orthonormal basis
    eigenvalues = S ** 2  # Shape: (n*K,)
    eigenvectors = Vh.T  # Shape: (n*K, p)


    # Update Fisher matrix
    # global_AtA += D 
    

    # Compute projection of A.T @ b onto the orthonormal basis E
    try:
        a_coeffs = torch.matmul(eigenvectors, At_b) # Shape: (p,)
    except:
        import pdb; pdb.set_trace()

    # Compute selection criterion: a_j^2 / lambda_j
    # Add small epsilon to eigenvalues to avoid division by zero
    lambda_reg = eigenvalues + 1e-6
    criterion = (a_coeffs ** 2) / lambda_reg  # Shape: (p,)
    
    # Select top r indices
    _, top_r_indices = torch.topk(criterion, low_rank, largest=True)
    
    # Extract selected eigenvalues and eigenvectors
    selected_eigenvalues = eigenvalues[top_r_indices]  # Shape: (r,)
    selected_eigenvectors = eigenvectors[top_r_indices, :]  # Shape: (r, p)
    
    # Compute weight update: E(r) Lambda(r)^-1 E(r)^T b
    try:
        Et_At_b = torch.matmul(selected_eigenvectors, At_b)  # Shape: (r,)
    except:
        import pdb; pdb.set_trace()
    lambda_r_inv = 1.0 / (selected_eigenvalues + 1e-6)  # Shape: (r,)
    w_update_flat = torch.matmul(selected_eigenvectors.T, lambda_r_inv * Et_At_b)  # Shape: (p,)

    # Apply update
    weights_updated = []
    offset = 0
    for shape in param_shapes:
        numel = torch.prod(torch.tensor(shape)).item()
        param_update = w_update_flat[offset:offset + numel].reshape(shape)
        weights_updated.append(param_update)
        offset += numel
    
    with torch.no_grad():
        for param, update in zip(model.parameters(), weights_updated):
            if param.requires_grad:
                param.copy_(update)

    return model