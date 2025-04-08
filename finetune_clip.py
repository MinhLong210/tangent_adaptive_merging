r"""
this script finetunes CLIP model form huggingface https://huggingface.co/openai/clip-vit-large-patch14

## Use with Transformers

```python
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
```
"""
# %%
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
import os
import csv

log = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("medium")


def setup_fabric(cfg: DictConfig):
    from lightning.fabric.loggers.tensorboard import TensorBoardLogger

    logger = TensorBoardLogger(
        root_dir=Path("logs_l2_loss") / cfg.model_name / cfg.dataset_name,
        name=cfg.finetuning_mode,
    )
    fabric = L.Fabric(**cfg.fabric, loggers=logger)
    fabric.launch()
    return fabric


def _get_submodules(model: nn.Module, key):
    """
    Retrieves the parent module, target module, and target module name for a given key in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to retrieve submodules from.
        key (str): The key representing the submodule to retrieve.

    Returns:
        Tuple[nn.Module, nn.Module, str]: A tuple containing the parent module, target module, and target module name.
    """
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def linearize_lora_model(model: nn.Module):
    """
    Linearizes the LoraLayer modules in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be linearized.

    Returns:
        nn.Module: The linearized PyTorch model.
    """
    for key, module in model.named_modules():
        if isinstance(module, LoraLayer) and isinstance(module, nn.Linear):
            log.debug(f"convert {key} to linearized lora layer")
            parent, target, target_name = _get_submodules(model, key)
            setattr(parent, target_name, LinearizedModelWraper(target))
    return model


def load_clip_processor_and_model(
    model_name_or_path: str,
    lora_config=None,
    linearized_lora: bool = False,
    local_files_only=True,
    random_seed: int = 42,
):
    L.seed_everything(random_seed)
    with TitledLog(" Create model ", log_fn=log.info):
        assert (
            model_name_or_path in CLIP_MODELS
        ), f"Unknown model name or path: {model_name_or_path}"
        processor, clip_model = load_clip_model(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        clip_model = freeze_unless_image_model(clip_model)
        clip_vision_model = clip_model.vision_model
        clip_text_model = clip_model.text_model

        if lora_config is not None:
            lora_config = instantiate(lora_config)
            lora_vision_model = get_peft_model(clip_vision_model, lora_config)
            if linearized_lora:
                lora_vision_model = linearize_lora_model(lora_vision_model)
            lora_vision_model.print_trainable_parameters()
            clip_vision_model = lora_vision_model

    clip_model.vision_model = clip_vision_model
    clip_model.text_model = clip_text_model
    return processor, clip_model, clip_vision_model, clip_text_model


# %%
@hydra.main(
    config_path="config",
    config_name=None,
    version_base=None,
)
def main(cfg: DictConfig):
    setup_colorlogging(force=True)

    fabric = setup_fabric(cfg)
    if fabric.logger is not None:
        # save `cfg` to fabric.logger.log_dir/config.yaml
        if not os.path.exists(fabric.logger.log_dir):
            os.makedirs(fabric.logger.log_dir)
        config_path = os.path.join(fabric.logger.log_dir, "config.yaml")
        OmegaConf.save(cfg, config_path)

    # create model
    (
        clip_processor,
        clip_model,
        clip_vision_model,
        clip_text_model,
    ) = load_clip_processor_and_model(
        cfg.model.model_name_or_path,
        cfg.lora_config,
        linearized_lora=cfg.linearized_lora,
        random_seed=cfg.seed,
    )
    fabric.setup_module(clip_model.visual_projection)
    clip_vision_model = fabric.setup_module(clip_vision_model)

    # setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in clip_vision_model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    lr_scheduler = CosineAnnealingWithWarmup(
        optimizer,
        base_lrs=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
    )

    # load data
    with TitledLog(" Load data ", log_fn=log.info):
        assert (
            cfg.model.batch_size % cfg.fabric.devices == 0
        ), "batch_size must be divisible by devices"
        cfg.batch_size = cfg.model.batch_size // cfg.fabric.devices
        input_size = cfg.model.input_size
        datamodule: pl.LightningDataModule = instantiate(
            cfg.datamodule,
            train_transform=transforms.Compose(
                [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
            ),
            test_transform=transforms.Compose(
                [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
            ),
        )
        train_loader = datamodule.train_dataloader()
        test_loader = datamodule.test_dataloader()
        print("training dataset", train_loader.dataset)
        print("test dataset", test_loader.dataset)

        train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # precompute the text features
    text = [f"a photo of a {c}" for c in datamodule.classes]
    text_input = clip_processor(text, return_tensors="pt", padding=True)
    text_embeds = clip_model.get_text_features(**text_input)

    # finetuning
    step_idx = 0
    clip_vision_model.train()
    csv_file = os.path.join(fabric.logger.log_dir, "train_stats.csv")
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step_idx", "loss", "learning_rate", "train_accuracy", "test_accuracy"])

    while step_idx < cfg.max_steps:
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            images, labels = batch
            image_embeds = clip_model.get_image_features(pixel_values=images)

            # Normalize features
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds.to(image_embeds.device)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            # Cosine similarity as logits
            logit_scale = clip_model.logit_scale.exp().item()
            logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
            logits_per_image = logits_per_text.t()

            # loss = F.cross_entropy(logits_per_image, labels)

            labels_ = F.one_hot(labels, num_classes=logits_per_image.shape[1]).float()
            loss = F.mse_loss(logits_per_image.softmax(dim=1), labels_) # Use quadratic regression loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step(step_idx)
            step_idx += 1

            # Compute train accuracy
            train_accuracy = logits_per_image.argmax(dim=1).eq(labels).float().mean().item()

            # Compute test accuracy every 100 steps
            if step_idx % 100 == 0:
                clip_model.eval()
                test_correct, test_total = 0, 0
                with torch.no_grad():
                    for test_images, test_labels in test_loader:
                        test_image_embeds = clip_model.get_image_features(pixel_values=test_images)
                        test_image_embeds = test_image_embeds / test_image_embeds.norm(p=2, dim=-1, keepdim=True)

                        test_logits = torch.matmul(text_embeds, test_image_embeds.t()) * logit_scale
                        test_predictions = test_logits.t().argmax(dim=1)
                        test_correct += (test_predictions == test_labels).sum().item()
                        test_total += test_labels.size(0)

                test_accuracy = test_correct / test_total
                clip_model.train()
            else:
                test_accuracy = None  # Skip logging test accuracy for other iterations

            print(
                f"[step_idx:{step_idx}] loss: {loss.item():.2f}, "
                f"lr: {lr_scheduler.get_lrs()[0]:.2e}, "
                f"train acc: {train_accuracy:.2f}, test acc: {test_accuracy:.2f}" if test_accuracy is not None else ""
            )

            # Log training stats to CSV
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([step_idx, loss.item(), lr_scheduler.get_lrs()[0], train_accuracy, test_accuracy or "NA"])


if __name__ == "__main__":
    main()
