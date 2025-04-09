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
from peta.models.clip import (
    CLIP_MODELS,
    freeze_unless_image_model,
    get_lora_vision_model,
    load_clip_model,
)
from peta.models.LinearizedModel import LinearizedModelWraper
from peta.optim import CosineAnnealingWithWarmup
from peta.utils.logging import TitledLog, setup_colorlogging

import copy
import time
from torch.func import jacrev, vmap

def closed_form_linear_clip(clip_model, clip_processor, train_loader, text, text_embeds, config):
    """
    Args:
        clip_model: Huggingface CLIP model (or PeftModel wrapping a CLIP model).
        clip_processor: Processor of CLIP model.
        text: List of text captions (length K for K classes).
        text_embeds: Precomputed text embeddings.
        config: Configuration object containing lora_config, slice_size, etc.
    """
    # Get a sample batch to determine device
    for batch_idx, batch in enumerate(train_loader):
        images_temp, labels = batch
        break

    # Move model to device and set to eval mode
    clip_model = clip_model.to(images_temp.device)
    clip_model.eval()

    # Initialize new CLIP model
    updated_clip_model = copy.deepcopy(clip_model)

    # Precompute text embeddings normalization and logit scale
    text_embeds = text_embeds.to(images_temp.device)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    logit_scale = clip_model.base_model.logit_scale.exp() if hasattr(clip_model, 'base_model') else clip_model.logit_scale.exp()

    # Define the forward function for a single sample (taking a tensor directly)
    def single_model_forward(param_slice, pixel_values, text_embeds):
        # pixel_values: Shape (channels, height, width)
        inputs = {"pixel_values": pixel_values.unsqueeze(0)}  # Add batch dim: (1, channels, height, width)
        state_dict = clip_model.vision_model.state_dict()
        full_param = state_dict[name].clone()
        full_param[slice_start:slice_end] = param_slice.reshape(slice_size, full_param.shape[1])
        state_dict[name] = full_param

        from torch.func import functional_call
        vision_outputs = functional_call(clip_model.vision_model, state_dict, (inputs["pixel_values"],))
        image_embeds = vision_outputs.pooler_output  # Shape: (1, hidden_size)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        if hasattr(clip_model, 'base_model'):
            image_embeds = clip_model.base_model.visual_projection(image_embeds)
        else:
            image_embeds = clip_model.visual_projection(image_embeds)
        
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()  # Shape: (1, num_classes)
        return logits_per_image.softmax(dim=1).squeeze(0)  # Shape: (num_classes,)

    # Iterate over parameters
    for (name, param), (name_clone, param_clone) in zip(
        clip_model.vision_model.named_parameters(), 
        updated_clip_model.vision_model.named_parameters()
    ):
        for target in config.lora_config.target_modules:
            if target in name and "lora" not in name and "bias" not in name:
                print(name)
                slice_size = config.slice_size
                num_slices = param.shape[0] // slice_size
                slice_param_size = slice_size * param.shape[1]

                for slice_idx in range(num_slices):
                    start_time = time.time()
                    print(f"Slice: {slice_idx+1}/{num_slices}")
                    slice_start = slice_idx * slice_size
                    slice_end = slice_start + slice_size

                    # Define sketch size and sketching matrix
                    sketch_size = min(1000, slice_param_size // 10)
                    sketching_matrix = torch.randn(slice_param_size, sketch_size).to(images_temp.device) / (sketch_size ** 0.5)

                    # Initialize in sketched space
                    global_At_A_sketch = torch.zeros(sketch_size, sketch_size).to(images_temp.device)
                    global_At_b_sketch = torch.zeros(sketch_size).to(images_temp.device)

                    # global_At_A_sketch = torch.zeros(slice_param_size, slice_param_size).to(images_temp.device)
                    # global_At_b_sketch = torch.zeros(slice_param_size).to(images_temp.device)

                    for batch_idx, batch in enumerate(tqdm(train_loader)):
                        images, labels = batch
                        inputs = clip_processor(text=text, images=images, return_tensors="pt", padding=True).to(images.device)
                        pixel_values = inputs["pixel_values"]  # Shape: (batch_size, channels, height, width)
                        labels = F.one_hot(labels.squeeze(), num_classes=text_embeds.shape[0]).float().to(images.device)
                        batch_size, output_dim = labels.shape  # output_dim = num_classes

                        with torch.no_grad():
                            outputs = clip_model(**inputs)
                            logits = outputs.logits_per_image.softmax(dim=1).detach()

                        # Compute sketched Jacobian using vmap
                        param_slice = param[slice_start:slice_end].reshape(-1)  # Shape: (slice_param_size,)
                        jacobian_fn = jacrev(single_model_forward, argnums=0)
                        jacobian_vmap = vmap(jacobian_fn, in_dims=(None, 0, None))  # Vectorize over batch dim (0) of pixel_values (second parameter of jacobian_fn)
                        sketched_jacobian = jacobian_vmap(param_slice, pixel_values, text_embeds) @ sketching_matrix  # Shape: (batch_size, num_classes, sketch_size)
                        A_matrix_sketch = sketched_jacobian.reshape(batch_size * output_dim, sketch_size)  # Shape: (batch_size * num_classes, sketch_size)

                    
                        # Compute b_vector
                        b_vector = (logits - labels).flatten()  # Shape: (batch_size * num_classes,)

                        # Accumulate in sketched space
                        global_At_A_sketch.add_(A_matrix_sketch.T @ A_matrix_sketch)
                        global_At_b_sketch.add_(A_matrix_sketch.T @ b_vector)


                        # Clean up
                        # Delete all intermediate tensors
                        del A_matrix_sketch, b_vector, sketched_jacobian, param_slice, jacobian_fn
                        del inputs, labels, logits, outputs
                        # Set variables to None to ensure no references persist
                        A_matrix_sketch, b_vector, sketched_jacobian, param_slice, jacobian_fn = None, None, None, None, None
                        inputs, labels, logits, outputs = None, None, None, None
                        # Clear the computational graph by detaching tensors (if any still require gradients)
                        if global_At_A_sketch.requires_grad:
                            global_At_A_sketch = global_At_A_sketch.detach()
                        if global_At_b_sketch.requires_grad:
                            global_At_b_sketch = global_At_b_sketch.detach()
                        # Clear GPU cache
                        torch.cuda.empty_cache()

                    # Solve the system in the sketched space
                    reg_term = 1e-5 * torch.eye(sketch_size).to(images_temp.device)
                    global_At_A_sketch_reg = global_At_A_sketch + reg_term
                    w_update_sketch = torch.linalg.solve(global_At_A_sketch_reg, global_At_b_sketch)  # Shape: (sketch_size,)

                    # Map back to full parameter space
                    w_update = sketching_matrix @ w_update_sketch  # Shape: (slice_param_size,)
                    w_update = w_update.reshape(slice_size, param.shape[1])  # Shape: (slice_size, param.shape[1])

                    # Update the parameter in the cloned model
                    with torch.no_grad():
                        if param_clone.data[slice_start:slice_end].shape == w_update.shape:
                            param_clone.data[slice_start:slice_end] += w_update
                        else:
                            print(f"Shape mismatch: {param_clone.data[slice_start:slice_end].shape} vs {w_update.shape}")

                    # Clean up
                    del global_At_A_sketch, global_At_b_sketch, w_update
                    torch.cuda.empty_cache()

                    print(f"Slice {slice_idx+1} completed. Time taken: {time.time() - start_time:.2f} seconds")

                    # After processing all batches for this slice, solve the system
                    # Add small regularization to ensure numerical stability
                    # reg_term = 1e-5 * torch.eye(global_At_A_sketch.shape[0]).to(images_temp.device)
                    # global_At_A_reg = global_At_A_sketch + reg_term
                    
                    # # Compute eigendecomposition
                    # eigenvalues, eigenvectors = torch.linalg.eigh(global_At_A_reg)
                    
                    # # Sort eigenvalues and eigenvectors in descending order
                    # idx = eigenvalues.argsort(descending=True)
                    # eigenvalues = eigenvalues[idx]
                    # eigenvectors = eigenvectors[:, idx]
                    
                    # if config.target_rank > 0:
                    #     # Project At_b onto eigenvectors
                    #     a_coeff = eigenvectors.T @ global_At_b_sketch
                        
                    #     # Selection criterion: a_coeff^2 / eigenvalues
                    #     selection_criterion = (a_coeff ** 2) / eigenvalues
                        
                    #     # Sort by selection criterion in descending order
                    #     sorted_indices = torch.argsort(selection_criterion, descending=True)

                    #     # Greedily select eigenvectors based on sorted criterion
                    #     target_rank = config.target_rank if hasattr(config, 'target_rank') else min(8, num_slices)
                    #     cumulative_rank = 0
                    #     selected_indices = []

                    #     for idx in sorted_indices:
                    #         # Add the eigenvector corresponding to this index
                    #         selected_indices.append(idx.item())

                    #         # Compute temporary solution with the selected eigenvectors
                    #         E_t_temp = eigenvectors[:, selected_indices]
                    #         S_t_inv_temp = torch.diag(1.0 / eigenvalues[selected_indices])
                    #         temp_solution = E_t_temp @ S_t_inv_temp @ (E_t_temp.T @ global_At_b_sketch)

                    #         # Reshape to check rank
                    #         temp_matrix = temp_solution.reshape(slice_size, param.shape[1])
                    #         rank = torch.linalg.matrix_rank(temp_matrix)

                    #         cumulative_rank += rank
                    #         if cumulative_rank >= target_rank:
                    #             break
                        
                    #     # Compute final closed-form solution with selected components
                    #     E_t = eigenvectors[:, selected_indices]  # Selected eigenvectors
                    #     S_t_inv = torch.diag(1.0/eigenvalues[selected_indices])  # Inverse of selected eigenvalues
                        
                    #     # w_update = E_t @ S_t^-1 @ E_t^T @ global_At_b
                    #     w_update = E_t @ S_t_inv @ (E_t.T @ global_At_b_sketch)
                    
                    # else:
                    #     # import pdb; pdb.set_trace()
                    #     w_update = eigenvectors @ torch.diag(1.0/eigenvalues) @ (eigenvectors.T @ global_At_b_sketch)
                    
                    # # Reshape to match parameter dimensions
                    # w_update = w_update.reshape(slice_size, param.shape[1])
                    
                    # # Update the parameter in the cloned model
                    # with torch.no_grad():
                    #     if param_clone.data[slice_start:slice_end].shape == w_update.shape:
                    #         param_clone.data[slice_start:slice_end] += w_update
                    #     else:
                    #         print(f"Shape mismatch: {param_clone.data[slice_start:slice_end].shape} vs {w_update.shape}")

                    #     # param_clone.data[slice_start:slice_end] +=0
                    
                    # # Clean up
                    # del global_At_A_sketch, global_At_b_sketch, eigenvalues, eigenvectors, w_update
                    # torch.cuda.empty_cache()

                    # if config.target_rank > 0:
                    #     print(f"Slice {slice_idx+1} completed. Final rank: {cumulative_rank}")
                    # else:
                    #     print("Full rank solution")
                    # print(f"Time taken: {time.time() - start_time:.2f} seconds")

    return updated_clip_model