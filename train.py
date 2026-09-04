"""Train the RamiGlyph dual-branch self-supervised model."""

import copy
import json
import os
import random
import sys

import numpy as np
import torch
import torch_geometric.transforms as T
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from augmentation.augment_node_position import Augment_node_position
from augmentation.reduce_node import RanDomReduceNodes
from dataloader.microglia import MicrogliaDataset
from model.dual_branch_model import dual_branch_swav_loss
from model.topo_utils import compute_topological_features_single
from model.train_utils import (
    build_lr_schedule,
    build_model,
    create_csv_logger,
    make_ckpt_dir,
    maybe_resume_from_checkpoint,
    run_validation,
)


def set_seed(seed: int = 42):
    """Seed all random number generators used during training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def batch_to_device(batch, device):
    """Move both graph views and topological views to the target device."""
    (graph1, graph2), (topo1, topo2) = batch
    return (
        (graph1.to(device), graph2.to(device)),
        (topo1.to(device), topo2.to(device)),
    )


class GraphTransformCompose:
    """Apply the configured graph augmentations in sequence."""

    def __init__(
        self,
        keep_nodes=1000,
        drop_branch=0,
        jitter_scale=0.025,
        trans_scale=1,
        rota=True,
        flip=True,
        flip_prob=0.5,
        soma_id=0,
        seed=None,
    ):
        self.transforms = [
            RanDomReduceNodes(
                keep_node=keep_nodes,
                n_branch=drop_branch,
                soma_id=soma_id,
                seed=seed,
            ),
            Augment_node_position(
                jitter_scale=jitter_scale,
                trans_scale=trans_scale,
                rota=rota,
                flip=flip,
                flip_prob=flip_prob,
                seed=seed,
            ),
            T.AddRandomWalkPE(walk_length=20, attr_name="pe"),
        ]

    def __call__(self, sample):
        data = sample
        for transform in self.transforms:
            data = transform(data)
        return data


class MultiViewDataInjector:
    """Generate augmented graph views and attach topological features."""

    def __init__(self, transformers, topo_config=None):
        self.transforms = transformers
        self.topo_config = topo_config or {}
        self.resolution = self.topo_config.get("resolution", 50)
        self.soma_node_id = self.topo_config.get("soma_node_id", 0)

    def __call__(self, sample):
        augmented_views = []
        for transform in self.transforms:
            augmented_graph = transform(copy.deepcopy(sample))
            topo_image = compute_topological_features_single(
                augmented_graph,
                resolution=self.resolution,
                soma_node_id=self.soma_node_id,
            )
            augmented_graph.topo_feature = torch.FloatTensor(topo_image)
            augmented_views.append(augmented_graph)
        return augmented_views


def collate_fn_worker(batch):
    """Collate two graph views and their topological feature maps."""
    if len(batch) == 0:
        raise ValueError("Empty batch received")

    view1_graphs, view1_topos = [], []
    view2_graphs, view2_topos = [], []

    for sample in batch:
        graph1, graph2 = sample
        view1_graphs.append(graph1)
        view1_topos.append(graph1.topo_feature)
        view2_graphs.append(graph2)
        view2_topos.append(graph2.topo_feature)

    graph_batch1 = Batch.from_data_list(view1_graphs)
    graph_batch2 = Batch.from_data_list(view2_graphs)
    topo_batch1 = torch.stack(view1_topos).unsqueeze(1)
    topo_batch2 = torch.stack(view2_topos).unsqueeze(1)

    return (graph_batch1, graph_batch2), (topo_batch1, topo_batch2)


def load_config():
    """Load the training configuration stored beside this script."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.json"
    )
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)

    print("=" * 60)
    print("RamiGlyph Training")
    print("=" * 60)
    print("Loaded config from:", config_path)
    return config


def prepare_device():
    """Select the available training device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print("CUDA device name:", torch.cuda.get_device_name())
    return device


def prepare_transforms(config):
    """Build the two augmentation pipelines used for paired views."""
    print("\nPreparing transforms...")
    aug_cfg = config.get("augment", {})
    seed = config.get("seed", 42)

    shared = dict(
        keep_nodes=aug_cfg.get("keep_nodes", 500),
        drop_branch=aug_cfg.get("drop_branch", 10),
        jitter_scale=aug_cfg.get("jitter_scale", 0.5),
        flip=aug_cfg.get("flip", True),
        flip_prob=aug_cfg.get("flip_prob", 0.5),
        rota=aug_cfg.get("rotate", True),
    )
    t1 = GraphTransformCompose(
        **shared,
        trans_scale=aug_cfg.get("trans_scale_1", 10),
        seed=seed,
    )
    t2 = GraphTransformCompose(
        **shared,
        trans_scale=aug_cfg.get("trans_scale_2", 5),
        seed=seed + 1,
    )
    return t1, t2


def load_dataset(config, transformer_view_1, transformer_view_2):
    """Load the training and validation datasets with paired augmentations."""
    dataset_root = os.path.join(
        config["dataset"]["data_root"], config["dataset"]["dataset_name"]
    )
    print("\nData path:", dataset_root)

    topo_cfg = config.get("topological", {})

    def make_injector():
        return MultiViewDataInjector(
            [transformer_view_1, transformer_view_2], topo_config=topo_cfg
        )

    train_data = MicrogliaDataset(
        dataset_root, split="train", transform=make_injector()
    )
    val_data = MicrogliaDataset(dataset_root, split="val", transform=make_injector())
    print(f"\nTrain samples: {len(train_data)}, Val samples: {len(val_data)}")
    return train_data, val_data


def train_step(model, optimizer, scaler, batch, step, lr_schedule, config, device):
    """Run one optimization step and return detached loss values."""
    model.train()

    swav_cfg = config.get("swav", {})
    loss_weights = config.get("loss_weights", {"struct": 0.5, "topo": 0.5})
    freeze_prototypes_steps = swav_cfg.get("freeze_prototypes_steps", 100)
    use_amp = config.get("use_amp", True) and device.type == "cuda"

    if step < len(lr_schedule["struct"]):
        lr_struct = float(lr_schedule["struct"][step])
        lr_topo = float(lr_schedule["topo"][step])
        optimizer.param_groups[0]["lr"] = lr_struct
        optimizer.param_groups[1]["lr"] = lr_struct
        optimizer.param_groups[2]["lr"] = lr_topo
        optimizer.param_groups[3]["lr"] = lr_topo
    else:
        lr_struct = optimizer.param_groups[0]["lr"]
        lr_topo = optimizer.param_groups[2]["lr"]

    model.normalize_prototypes()
    (graph1, graph2), (topo1, topo2) = batch_to_device(batch, device)
    optimizer.zero_grad()

    with autocast(enabled=use_amp):
        _, _, out_struct_1, out_topo_1, _ = model(graph1, topo1)
        _, _, out_struct_2, out_topo_2, _ = model(graph2, topo2)
        total_loss, loss_struct, loss_topo = dual_branch_swav_loss(
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

    scaler.scale(total_loss).backward()

    if step < freeze_prototypes_steps:
        for name, parameter in model.named_parameters():
            if "prototypes" in name:
                parameter.grad = None

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()

    return (
        total_loss.detach(),
        loss_struct.detach(),
        loss_topo.detach(),
        lr_struct,
        lr_topo,
    )


def train_loop(model, optimizer, train_data, val_data, device, config, ckpt_dir):
    """Run step-based training, validation, logging, and checkpointing."""
    train_cfg = config["train"]
    swav_cfg = config.get("swav", {})
    loader_cfg = config.get("loader", {})
    seed = config.get("seed", 42)
    num_workers = loader_cfg.get("num_workers", 0)
    use_amp = config.get("use_amp", True) and device.type == "cuda"

    csv_file, csv_writer, csv_path = create_csv_logger(ckpt_dir, config)

    def worker_init_fn(worker_id):
        w_seed = seed + worker_id
        np.random.seed(w_seed)
        random.seed(w_seed)
        torch.manual_seed(w_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    loader_kwargs = dict(
        batch_size=train_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=loader_cfg.get("pin_memory", True),
        persistent_workers=(
            loader_cfg.get("persistent_workers", True) if num_workers > 0 else False
        ),
        prefetch_factor=(
            loader_cfg.get("prefetch_factor", 2) if num_workers > 0 else None
        ),
        collate_fn=collate_fn_worker,
        worker_init_fn=worker_init_fn,
    )
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=g,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)

    lr_schedule = build_lr_schedule(config)
    total_steps = train_cfg["total_steps"]
    log_interval = train_cfg.get("log_interval", 10)
    val_interval = train_cfg.get("val_interval", 500)
    checkpoint_freq = train_cfg.get("checkpoint_freq", 100)

    print(f"\nTotal training steps: {total_steps}")
    print(f"Warmup steps: {train_cfg.get('warmup_steps', 1000)}")
    print(f"AMP mixed precision: {'enabled' if use_amp else 'disabled'}")
    print(f"Log interval: every {log_interval} steps")

    scaler = GradScaler(enabled=use_amp)
    start_step, min_train_loss = maybe_resume_from_checkpoint(
        model, optimizer, ckpt_dir, device, scaler
    )

    print("\n" + "=" * 60)
    print("Starting RamiGlyph training (step-based)")
    print(f"Total steps: {total_steps}, Batch size: {train_cfg['batch_size']}")
    print(
        f"SwAV config: prototypes={swav_cfg.get('nmb_prototypes', 100)}, "
        f"temperature={swav_cfg.get('temperature', 0.1)}, "
        f"epsilon={swav_cfg.get('epsilon', 0.05)}"
    )
    print("=" * 60)

    def infinite_loader(loader):
        while True:
            yield from loader

    data_iter = infinite_loader(train_loader)
    step_buffer = []

    for step in range(start_step, total_steps):
        batch = next(data_iter)
        loss_t, loss_struct_t, loss_topo_t, lr_struct, lr_topo = train_step(
            model, optimizer, scaler, batch, step, lr_schedule, config, device
        )
        step_buffer.append(
            (step, loss_t, loss_struct_t, loss_topo_t, lr_struct, lr_topo)
        )

        if (step + 1) % log_interval == 0 or step == total_steps - 1:
            resolved = [
                (s, lt.item(), lst.item(), ltt.item(), lrs, lrt)
                for s, lt, lst, ltt, lrs, lrt in step_buffer
            ]
            step_buffer.clear()

            for s, loss, loss_struct, loss_topo, lrs, lrt in resolved:
                csv_writer.writerow(
                    [
                        s + 1,
                        total_steps,
                        f"{loss:.6f}",
                        f"{loss_struct:.6f}",
                        f"{loss_topo:.6f}",
                        "",
                        "",
                        "",
                        f"{lrs:.6e}",
                        f"{lrt:.6e}",
                    ]
                )
            csv_file.flush()

            for s, loss, loss_struct, loss_topo, lrs, lrt in resolved:
                best_flag = ""
                if loss < min_train_loss:
                    min_train_loss = loss
                    best_flag = " | New Best!"
                    torch.save(
                        {
                            "step": s + 1,
                            "min_loss": min_train_loss,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scaler_state": scaler.state_dict(),
                        },
                        os.path.join(ckpt_dir, "best_checkpoint.pt"),
                    )
                print(
                    f"Step [{s + 1}/{total_steps}] Loss: {loss:.4f} "
                    f"(S:{loss_struct:.4f} T:{loss_topo:.4f}) "
                    f"LR_S: {lrs:.6f} LR_T: {lrt:.6f}{best_flag}"
                )

        if (step + 1) % val_interval == 0:
            val_loss, val_loss_struct, val_loss_topo = run_validation(
                model, val_loader, config, device, batch_to_device
            )
            _loss = loss_t.item()
            _loss_s = loss_struct_t.item()
            _loss_t = loss_topo_t.item()
            _lr_struct = optimizer.param_groups[0]["lr"]
            _lr_topo = optimizer.param_groups[2]["lr"]
            print(
                f"  -> Val Loss: {val_loss:.4f} "
                f"(S:{val_loss_struct:.4f} T:{val_loss_topo:.4f})"
            )
            csv_writer.writerow(
                [
                    step + 1,
                    step + 1,
                    f"{_loss:.6f}",
                    f"{_loss_s:.6f}",
                    f"{_loss_t:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_loss_struct:.6f}",
                    f"{val_loss_topo:.6f}",
                    f"{_lr_struct:.6e}",
                    f"{_lr_topo:.6e}",
                ]
            )
            csv_file.flush()

        if (step + 1) % checkpoint_freq == 0 or step == total_steps - 1:
            ckpt = {
                "step": step + 1,
                "min_loss": min_train_loss,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
            }
            torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint.pth.tar"))
            torch.save(ckpt, os.path.join(ckpt_dir, f"checkpoint_step_{step + 1}.pt"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  -> Checkpoint saved at step {step + 1}")

    print("\n" + "=" * 60)
    print("Training Completed!")
    print(f"Best loss: {min_train_loss:.6f}")
    print(f"Models saved in: {ckpt_dir}")
    print("=" * 60)
    csv_file.close()
    print(f"Training log saved to: {csv_path}")


def main():
    """Initialize and run RamiGlyph training."""
    config = load_config()
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"\nRandom seed set to: {seed}")

    device = prepare_device()
    t1, t2 = prepare_transforms(config)
    train_data, val_data = load_dataset(config, t1, t2)
    model, optimizer = build_model(config, device)
    ckpt_dir = make_ckpt_dir(config)
    train_loop(model, optimizer, train_data, val_data, device, config, ckpt_dir)


if __name__ == "__main__":
    main()
