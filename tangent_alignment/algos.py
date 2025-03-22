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
from torch.func import jacrev

def closed_form_linear_clip(clip_model, clip_processor, train_loader, text, config):
    ################################## SLICE IMPLEMENATION: ACCUMULATE THE AT^A, A^Tb ACROSS BATCH
    """
    Args:
        clip_model:(Huggingface CLIP model).
        clip_processor: Processor of CLIP model.
        text: List of text captions (length K for K classes).
        lora_config: LoRA configuration (LoraConfig object).
    
    """

    ######################################################## Step 1: Compute matrix A ###############################################
    # Define target modules typically used for LoRA (vision encoder only)
    target_modules = config.lora_config.target_modules
    
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        images_temp, labels = batch
        break

    # Move model to device
    clip_model = clip_model.to(images_temp.device)
    clip_model.eval() # Set to eval mode

    # Initialize new CLIP model
    updated_clip_model = copy.deepcopy(clip_model)
    

    for (name, param), (name_clone, param_clone) in zip(
        clip_model.vision_model.named_parameters(), 
        updated_clip_model.vision_model.named_parameters()
    ):  # Since LoRA is applied only to vision model
        for target in target_modules:
            if target in name and "lora" not in name and "bias" not in name:
                print(name)
                param.requires_grad_(True)

                # Slice params
                slice_size = config.slice_size
                num_slices = param.shape[0] // slice_size
                slice_param_size = slice_size * param.shape[1]  # 16 * 768 = 12,288

                # Loop over the slices
                for slice_idx in range(num_slices):
                    start_time = time.time()
                    print(f"Slice: {slice_idx+1}/{num_slices}")
                    slice_start = slice_idx * slice_size
                    slice_end = slice_start + slice_size

                    # Initialize accumulated terms for each slice
                    global_At_A = torch.zeros((slice_param_size, slice_param_size)).to(images_temp.device)
                    global_At_b = torch.zeros(slice_param_size).to(images_temp.device)       
                    
                    for batch_idx, batch in enumerate(tqdm(train_loader)):
                        images, labels = batch

                        # Process image and text for CLIP
                        inputs = clip_processor(text=text, images=images, return_tensors="pt", padding=True).to(images.device)

                        # One-hot encode labels
                        labels = labels.squeeze()  # Shape: (n,)
                        labels = F.one_hot(labels, num_classes=10).float()  # Shape: (n, K)

                        
                        batch_size, output_dim = labels.shape
                        # grad_output = torch.zeros_like(logits)
                        
                        # Batched computation of A matrix
                        # Forward pass of pretrained model
                        # with torch.no_grad():
                        #     outputs = clip_model(**inputs)
                        #     logits = outputs.logits_per_image  # Shape: (batch_size, num_classes)
                        # grads_list = []
                        # for b_idx in range(batch_size):
                        #     for i in range(output_dim):
                        #         grad_output = torch.zeros_like(logits[b_idx])
                        #         grad_output[i] = 1.0
                        #         grads = torch.autograd.grad(
                        #             logits[b_idx], param, grad_output, 
                        #             retain_graph=True  # Only retain if needed
                        #         )[0]
                        #         grads_list.append(grads[slice_start:slice_end, :].flatten().detach())
                        # grads = None
                        # # Construct A_matrix_slice from grads_list
                        # A_matrix_slice = torch.stack(grads_list, dim=0).to(images.device) # Shape: (n * K, p)

                        # Parallel ====> This gives 0 Jacobian
                        # Define the forward function for Jacobian computation
                        def model_forward(param, inputs):
                            # Get the current state dict of the model
                            state_dict = clip_model.vision_model.state_dict()
                            
                            # Update the specific parameter slice in a new state dict
                            full_param = state_dict[name].clone()  # Clone to avoid modifying the original
                            # full_param[slice_start:slice_end] = param_slice.reshape(slice_size, full_param.shape[1])
                            full_param = param
                            state_dict[name] = full_param

                            # Use functional_call to compute the forward pass without modifying the model in-place
                            from torch.func import functional_call
                            vision_outputs = functional_call(clip_model.vision_model, state_dict, (inputs["pixel_values"],))  # Pass pixel_values as a positional argument
                            # Extract the pooler output (or last hidden state) from vision_outputs
                            image_embeds = vision_outputs.pooler_output  # Shape: (batch_size, hidden_size)
                            
                            # Normalize the image embeddings (as done in CLIP's forward pass)
                            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                            
                            # Get the text embeddings (we can use the original model for this part since we're not modifying text parameters)
                            text_input = clip_processor(text, return_tensors="pt", padding=True).to(image_embeds.device)
                            text_embeds = clip_model.get_text_features(**text_input)
                            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                            
                            # Project the embeddings
                            image_embeds = clip_model.visual_projection(image_embeds)  # Shape: (batch_size, projection_dim)
                            text_embeds = clip_model.text_projection(text_embeds)  # Shape: (num_classes, projection_dim)
                            
                            # Compute logits_per_image
                            logit_scale = clip_model.logit_scale.exp()
                            logits_per_image = logit_scale * image_embeds @ text_embeds.t()  # Shape: (batch_size, num_classes)
                            
                            return logits_per_image  # Shape: (batch_size, num_classes)outputs = functional_call(clip_model.vision_model, state_dict, (), inputs["pixel_values"]).logits_per_image  # Shape: (batch_size, num_classes)
                            
                        # Compute the Jacobian: ∂logits/∂param
                        jacobian_fn = jacrev(model_forward, argnums=0)  # Differentiate w.r.t. param
                        jacobian = jacobian_fn(param, inputs)  # Shape: (batch_size * output_dim, param_shape)
                        A_matrix_slice = jacobian[:, :, slice_start:slice_end].reshape(batch_size * output_dim, -1)
                        import pdb; pdb.set_trace()


                        ######################################################## Step 2: Closed form solution ###############################################
                        # After accumulating A_matrix_slice for the batch
                        # Compute b_vector = logits - labels
                        b_vector = (logits - labels).flatten()  # Shape: (batch_size * num_classes)
                        
                        # Compute A^T A and A^T b for this batch
                        global_At_A.add_(A_matrix_slice.T @ A_matrix_slice)  # Shape: (p, p)
                        global_At_b.add_(A_matrix_slice.T @ b_vector)       # Shape: (p,)
                        
                        # Clean up
                        del A_matrix_slice, b_vector, jacobian
                        torch.cuda.empty_cache()
                
                    # After processing all batches for this slice, solve the system
                    # Add small regularization to ensure numerical stability
                    reg_term = 1e-5 * torch.eye(global_At_A.shape[0]).to(images_temp.device)
                    global_At_A_reg = global_At_A + reg_term
                    
                    # Compute eigendecomposition
                    eigenvalues, eigenvectors = torch.linalg.eigh(global_At_A_reg)
                    
                    # Sort eigenvalues and eigenvectors in descending order
                    idx = eigenvalues.argsort(descending=True)
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]
                    
                    # Project At_b onto eigenvectors
                    a_coeff = eigenvectors.T @ global_At_b
                    
                    # Selection criterion: a_coeff^2 / eigenvalues
                    selection_criterion = (a_coeff ** 2) / eigenvalues
                    
                    # Sort by selection criterion in descending order
                    sorted_indices = torch.argsort(selection_criterion, descending=True)

                    # Greedily select eigenvectors based on sorted criterion
                    target_rank = config.target_rank if hasattr(config, 'target_rank') else min(8, num_slices)
                    cumulative_rank = 0
                    selected_indices = []

                    for idx in sorted_indices:
                        # Add the eigenvector corresponding to this index
                        selected_indices.append(idx.item())

                        # Compute temporary solution with the selected eigenvectors
                        E_t_temp = eigenvectors[:, selected_indices]
                        S_t_inv_temp = torch.diag(1.0 / eigenvalues[selected_indices])
                        temp_solution = E_t_temp @ S_t_inv_temp @ (E_t_temp.T @ global_At_b)

                        # Reshape to check rank
                        temp_matrix = temp_solution.reshape(slice_size, param.shape[1])
                        import pdb; pdb.set_trace()
                        rank = torch.linalg.matrix_rank(temp_matrix)

                        cumulative_rank += rank
                        if cumulative_rank >= target_rank:
                            break
                    
                    # Compute final closed-form solution with selected components
                    E_t = eigenvectors[:, selected_indices]  # Selected eigenvectors
                    S_t_inv = torch.diag(1.0/eigenvalues[selected_indices])  # Inverse of selected eigenvalues
                    
                    # w_update = E_t @ S_t^-1 @ E_t^T @ global_At_b
                    w_update = E_t @ S_t_inv @ (E_t.T @ global_At_b)
                    
                    # Reshape to match parameter dimensions
                    w_update = w_update.reshape(slice_size, param.shape[1])
                    
                    # Update the parameter in the cloned model
                    with torch.no_grad():
                        if param_clone.data[slice_start:slice_end].shape == w_update.shape:
                            param_clone.data[slice_start:slice_end] += w_update
                        else:
                            print(f"Shape mismatch: {param_clone.data[slice_start:slice_end].shape} vs {w_update.shape}")
                    
                    # Clean up
                    del global_At_A, global_At_b, eigenvalues, eigenvectors, w_update
                    torch.cuda.empty_cache()
                    
                    print(f"Slice {slice_idx+1} completed. Final rank: {cumulative_rank}")
                    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # Return the updated model
    return updated_clip_model