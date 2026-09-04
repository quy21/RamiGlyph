"""Training, checkpoint, logging, and validation utilities."""

import csv
import gc
import math
import os
from datetime import datetime

import numpy as np
import torch
from torch.optim import AdamW

from .dual_branch_model import build_ramiglyph_model, dual_branch_swav_loss


def build_lr_schedule(config):
    """Build warmup and cosine-decay schedules for both branches."""
    train_cfg = config["train"]
    total_steps = train_cfg["total_steps"]
    warmup_steps = train_cfg.get("warmup_steps", 1000)
    lr_struct = train_cfg.get("lr_struct", 0.00005)
    lr_topo = train_cfg.get("lr_topo", 0.0002)
    min_lr_ratio = train_cfg.get("min_lr_ratio", 0.01)

    def build_schedule(base_lr):
        warmup_schedule = (
            np.linspace(0, base_lr, warmup_steps) if warmup_steps > 0 else np.array([])
        )
        cosine_steps = total_steps - warmup_steps
        min_lr = base_lr * min_lr_ratio
        cosine_schedule = np.array(
            [
                min_lr
                + (base_lr - min_lr)
                * 0.5
                * (1.0 + math.cos(math.pi * i / cosine_steps))
                for i in range(cosine_steps)
            ]
        )
        return np.concatenate((warmup_schedule, cosine_schedule))

    return {"struct": build_schedule(lr_struct), "topo": build_schedule(lr_topo)}


def build_model(config, device):
    """Build RamiGlyph and its branch-specific AdamW optimizer."""
    print("\nBuilding RamiGlyph model...")
    model = build_ramiglyph_model(config, device)
    total_params = sum(p.numel() for p in model.trainable_parameters())
    print(f"Model parameters: {total_params:,}")

    train_cfg = config["train"]
    optimizer = AdamW(
        [
            {
                "params": model.encoder.struct_encoder.parameters(),
                "lr": train_cfg["lr_struct"],
            },
            {
                "params": model.prototypes_struct.parameters(),
                "lr": train_cfg["lr_struct"],
            },
            {
                "params": model.encoder.topo_encoder.parameters(),
                "lr": train_cfg["lr_topo"],
            },
            {"params": model.prototypes_topo.parameters(), "lr": train_cfg["lr_topo"]},
        ],
        weight_decay=train_cfg["weight_decay"],
    )

    print(
        f'Learning rates: struct={train_cfg["lr_struct"]}, topo={train_cfg["lr_topo"]}'
    )
    return model, optimizer


def make_ckpt_dir(config):
    """Create and return the configured checkpoint directory."""
    ckpt_dir = config.get("checkpoint_dir", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print("\nCheckpoints will be saved to:", ckpt_dir)
    return ckpt_dir


def maybe_resume_from_checkpoint(model, optimizer, ckpt_dir, device, scaler=None):
    """Restore model, optimizer, and optional AMP scaler state."""
    best_ckpt = os.path.join(ckpt_dir, "checkpoint.pth.tar")
    if os.path.isfile(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scaler is not None and "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_step = checkpoint.get("step", checkpoint.get("epoch", 0))
        min_loss = checkpoint.get("min_loss", float("inf"))
        print(
            f"Resumed from checkpoint at step {start_step} (best loss {min_loss:.6f})"
        )
        return start_step, min_loss
    return 0, float("inf")


def create_csv_logger(ckpt_dir: str, config: dict):
    """Create the timestamped CSV training log."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(ckpt_dir, f"training_log_{timestamp}.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(["# RamiGlyph Training", "Timestamp: " + timestamp])
    csv_writer.writerow([])

    for section in [
        "model",
        "train",
        "augment",
        "topological",
        "loss_weights",
        "swav",
        "loader",
    ]:
        csv_writer.writerow([f"# {section.capitalize()}"])
        for key, value in config.get(section, {}).items():
            if not key.startswith("_"):
                csv_writer.writerow([key, value])
        csv_writer.writerow([])

    csv_writer.writerow(["# Training Data"])
    csv_writer.writerow(
        [
            "Step",
            "Total_Steps",
            "Train_Loss",
            "Train_Loss_Struct",
            "Train_Loss_Topo",
            "Val_Loss",
            "Val_Loss_Struct",
            "Val_Loss_Topo",
            "LR_Struct",
            "LR_Topo",
        ]
    )
    return csv_file, csv_writer, csv_path


def run_validation(model, val_loader, config, device, batch_to_device_fn):
    """Evaluate the dual-branch objective over a validation loader."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    model.eval()
    total_loss = total_loss_struct = total_loss_topo = 0.0
    count = 0

    swav_cfg = config.get("swav", {})
    loss_weights = config.get("loss_weights", {"struct": 0.5, "topo": 0.5})

    with torch.no_grad():
        for batch in val_loader:
            (graph1, graph2), (topo1, topo2) = batch_to_device_fn(batch, device)
            _, _, out_struct_1, out_topo_1, _ = model(graph1, topo1)
            _, _, out_struct_2, out_topo_2, _ = model(graph2, topo2)

            loss, loss_struct, loss_topo = dual_branch_swav_loss(
                out_struct_1,
                out_struct_2,
                out_topo_1,
                out_topo_2,
                temperature=swav_cfg.get("temperature", 0.1),
                epsilon=swav_cfg.get("epsilon", 0.05),
                sinkhorn_iterations=swav_cfg.get("sinkhorn_iterations", 3),
                w_struct=loss_weights["struct"],
                w_topo=loss_weights["topo"],
            )
            total_loss += loss.item()
            total_loss_struct += loss_struct.item()
            total_loss_topo += loss_topo.item()
            count += 1

    model.train()
    n = max(count, 1)
    return total_loss / n, total_loss_struct / n, total_loss_topo / n
