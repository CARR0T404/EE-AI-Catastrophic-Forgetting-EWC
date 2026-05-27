# ============================================================
# THREE SCENARIOS FOR CONTINUAL LEARNING — REAL SCALE-DOWN SIM v7 (FIXED FOR PAPER STYLE)
# ============================================================

import os
import json
import time
import math
import shutil
import random
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms

# ============================================================
# 1. CONFIG (LIGHT SCALE-DOWN)
# ============================================================

@dataclass
class CFG:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir: str = "three_scenarios_scaledown_REAL_outputs_v7"
    clean_output_dir: bool = True
    data_dir: str = "./data"
    download_mnist: bool = True

    # Scale-down runtime knobs for faster execution
    epochs_per_task: int = 2
    batch_size: int = 128
    num_workers: int = 2

    split_num_tasks: int = 5
    split_train_subset_per_task: int = 500
    split_test_subset_per_task: int = 200

    permuted_num_tasks: int = 5
    permuted_train_subset_per_task: int = 800
    permuted_test_subset_per_task: int = 300

    hidden_sizes: tuple = (256, 256)
    lr: float = 0.003
    optimizer: str = "adam"
    max_grad_norm: float = 50.0

    fisher_batches: int = 5
    ewc_lambda_grid: tuple = (0.0, 10.0, 1000.0, 100000.0)
    online_ewc_gamma_grid: tuple = (1.0, 0.8, 0.6)

    si_c_grid: tuple = (0.0, 0.1, 10.0, 1000.0)
    si_xi: float = 0.1

    xdg_percent_grid: tuple = (0, 20, 40, 60, 80)

    split_memory_budgets: tuple = (20, 100, 500, 2000)
    permuted_memory_budgets: tuple = (100, 500, 2000, 5000)

    run_figD_gridsearch: bool = True
    run_figC_memory_budget: bool = True
    run_split: bool = True
    run_permuted: bool = True
    dpi: int = 300

cfg = CFG()

if cfg.clean_output_dir and os.path.exists(cfg.output_dir):
    shutil.rmtree(cfg.output_dir)
os.makedirs(cfg.output_dir, exist_ok=True)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ============================================================
# 2. REPRODUCIBILITY & DATASETS
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)

SPLIT_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

class SplitMNISTTask(Dataset):
    def __init__(self, base_dataset, task_id: int, scenario: str):
        self.base_dataset = base_dataset
        self.task_id = task_id
        self.scenario = scenario
        self.pair = SPLIT_PAIRS[task_id]
        self.indices = []
        targets = np.asarray(base_dataset.targets)
        for i, y in enumerate(targets):
            if int(y) in self.pair:
                self.indices.append(i)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        x, y_digit = self.base_dataset[base_idx]
        y_digit = int(y_digit)
        if self.scenario in ["Task-IL", "Domain-IL"]:
            y = 0 if y_digit == self.pair[0] else 1
        else:
            y = y_digit
        return x, torch.tensor(y, dtype=torch.long), torch.tensor(self.task_id, dtype=torch.long)

class PermutedMNISTTask(Dataset):
    def __init__(self, base_dataset, task_id: int, permutation: torch.Tensor, scenario: str):
        self.base_dataset = base_dataset
        self.task_id = task_id
        self.permutation = permutation
        self.scenario = scenario

    def __len__(self): return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y_digit = self.base_dataset[idx]
        y_digit = int(y_digit)
        x = x.view(-1)[self.permutation].view(1, 28, 28)
        if self.scenario in ["Task-IL", "Domain-IL"]:
            y = y_digit
        else:
            y = self.task_id * 10 + y_digit
        return x, torch.tensor(y, dtype=torch.long), torch.tensor(self.task_id, dtype=torch.long)

def make_permutation(seed: int):
    rng = np.random.default_rng(seed)
    perm = np.arange(784)
    rng.shuffle(perm)
    return torch.tensor(perm, dtype=torch.long)

def subset_dataset(ds, subset_size, seed):
    rng = np.random.default_rng(seed)
    n = len(ds)
    size = min(subset_size, n)
    idx = np.arange(n)
    rng.shuffle(idx)
    return Subset(ds, idx[:size].tolist())

def build_loaders(protocol: str, scenario: str):
    transform = transforms.ToTensor()
    train_base = MNIST(cfg.data_dir, train=True, download=cfg.download_mnist, transform=transform)
    test_base = MNIST(cfg.data_dir, train=False, download=cfg.download_mnist, transform=transform)

    train_loaders, test_loaders = [], []

    if protocol == "split":
        num_tasks = cfg.split_num_tasks
        train_subset = cfg.split_train_subset_per_task
        test_subset = cfg.split_test_subset_per_task
        for task_id in range(num_tasks):
            tr_ds = SplitMNISTTask(train_base, task_id, scenario)
            te_ds = SplitMNISTTask(test_base, task_id, scenario)
            tr_sub = subset_dataset(tr_ds, train_subset, cfg.seed + 10 * task_id)
            te_sub = subset_dataset(te_ds, test_subset, cfg.seed + 1000 + 10 * task_id)
            train_loaders.append(DataLoader(tr_sub, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers))
            test_loaders.append(DataLoader(te_sub, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers))

    elif protocol == "permuted":
        num_tasks = cfg.permuted_num_tasks
        train_subset = cfg.permuted_train_subset_per_task
        test_subset = cfg.permuted_test_subset_per_task
        permutations = [make_permutation(cfg.seed + 2000 + t) for t in range(num_tasks)]
        for task_id in range(num_tasks):
            tr_ds = PermutedMNISTTask(train_base, task_id, permutations[task_id], scenario)
            te_ds = PermutedMNISTTask(test_base, task_id, permutations[task_id], scenario)
            tr_sub = subset_dataset(tr_ds, train_subset, cfg.seed + 20 * task_id)
            te_sub = subset_dataset(te_ds, test_subset, cfg.seed + 2000 + 20 * task_id)
            train_loaders.append(DataLoader(tr_sub, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers))
            test_loaders.append(DataLoader(te_sub, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers))

    return train_loaders, test_loaders

def task_classes_for(protocol):
    return 2 if protocol == "split" else 10

def output_dim_for(protocol, scenario):
    n_tasks = cfg.split_num_tasks if protocol == "split" else cfg.permuted_num_tasks
    if protocol == "split":
        if scenario == "Task-IL": return 2 * n_tasks
        elif scenario == "Domain-IL": return 2
        elif scenario == "Class-IL": return 10
    if protocol == "permuted":
        if scenario == "Task-IL": return 10 * n_tasks
        elif scenario == "Domain-IL": return 10
        elif scenario == "Class-IL": return 10 * n_tasks

# ============================================================
# 4. MODEL & LOSSES
# ============================================================
class MLP(nn.Module):
    def __init__(self, output_dim, hidden_sizes=(128, 128)):
        super().__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(784, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_out = nn.Linear(hidden_sizes[1], output_dim)

    def forward(self, x, masks=None, return_features=False):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        if masks is not None: h1 = h1 * masks[0].to(h1.device)
        h2 = F.relu(self.fc2(h1))
        if masks is not None: h2 = h2 * masks[1].to(h2.device)
        logits = self.fc_out(h2)
        if return_features: return logits, h2
        return logits

def make_model(protocol, scenario): return MLP(output_dim_for(protocol, scenario), cfg.hidden_sizes).to(cfg.device)
def make_optimizer(model): return torch.optim.Adam(model.parameters(), lr=cfg.lr)

def make_xdg_masks(num_tasks, hidden_sizes, gated_percent, seed):
    rng = np.random.default_rng(seed)
    masks = {}
    keep_prob = 1.0 - gated_percent / 100.0
    for task_id in range(num_tasks):
        task_masks = []
        for h in hidden_sizes:
            m = (rng.random(h) < keep_prob).astype(np.float32)
            task_masks.append(torch.tensor(m, dtype=torch.float32).view(1, -1))
        masks[task_id] = task_masks
    return masks

def task_il_loss(logits, y, task_ids, protocol):
    classes_per_task = task_classes_for(protocol)
    loss = torch.tensor(0.0, device=logits.device)
    total = 0
    for tid in task_ids.unique():
        tid_int = int(tid.item())
        mask = task_ids == tid
        start, end = tid_int * classes_per_task, (tid_int + 1) * classes_per_task
        loss = loss + F.cross_entropy(logits[mask, start:end], y[mask], reduction="sum")
        total += int(mask.sum().item())
    return loss / max(total, 1)

def classification_loss(model, x, y, task_ids, protocol, scenario, xdg_masks=None):
    if xdg_masks is not None and scenario == "Task-IL":
        losses, weights = [], []
        for tid in task_ids.unique():
            mask = task_ids == tid
            tid_int = int(tid.item())
            logits = model(x[mask], masks=xdg_masks[tid_int])
            loss = task_il_loss(logits, y[mask], task_ids[mask], protocol) if scenario == "Task-IL" else F.cross_entropy(logits, y[mask])
            losses.append(loss * mask.sum())
            weights.append(mask.sum())
        return torch.stack(losses).sum() / torch.stack(weights).sum().clamp(min=1)
    logits = model(x)
    if scenario == "Task-IL": return task_il_loss(logits, y, task_ids, protocol)
    return F.cross_entropy(logits, y)

@torch.no_grad()
def evaluate_network(model, loader, protocol, scenario, xdg_masks=None):
    model.eval()
    total, correct = 0, 0
    classes_per_task = task_classes_for(protocol)
    for x, y, task_ids in loader:
        x, y, task_ids = x.to(cfg.device), y.to(cfg.device), task_ids.to(cfg.device)
        preds = torch.zeros_like(y)
        if xdg_masks is not None and scenario == "Task-IL":
            for tid in task_ids.unique():
                mask = task_ids == tid
                tid_int = int(tid.item())
                logits = model(x[mask], masks=xdg_masks[tid_int])
                start, end = tid_int * classes_per_task, (tid_int + 1) * classes_per_task
                preds[mask] = logits[:, start:end].argmax(dim=1)
        else:
            logits = model(x)
            if scenario == "Task-IL":
                for tid in task_ids.unique():
                    mask = task_ids == tid
                    tid_int = int(tid.item())
                    start, end = tid_int * classes_per_task, (tid_int + 1) * classes_per_task
                    preds[mask] = logits[mask, start:end].argmax(dim=1)
            else:
                preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

@torch.no_grad()
def evaluate_all_tasks(model, test_loaders, protocol, scenario, xdg_masks=None):
    return [evaluate_network(model, loader, protocol, scenario, xdg_masks) for loader in test_loaders]

# ============================================================
# 5. CONTINUAL LEARNING ALGORITHMS & MEMORY
# ============================================================
def copy_params(model): return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

def compute_fisher(model, loader, protocol, scenario, xdg_masks=None):
    model.eval()
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
    used = 0
    for batch_idx, (x, y, task_ids) in enumerate(loader):
        if batch_idx >= cfg.fisher_batches: break
        x, y, task_ids = x.to(cfg.device), y.to(cfg.device), task_ids.to(cfg.device)
        model.zero_grad()
        loss = classification_loss(model, x, y, task_ids, protocol, scenario, xdg_masks)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
        used += 1
    for name in fisher: fisher[name] /= max(used, 1)
    return fisher

def ewc_penalty(model, ewc_memory, ewc_lambda):
    if not ewc_memory or ewc_lambda <= 0: return torch.tensor(0.0, device=cfg.device)
    penalty = torch.tensor(0.0, device=cfg.device)
    for mem in ewc_memory:
        theta_star, fisher = mem["theta_star"], mem["fisher"]
        for name, p in model.named_parameters():
            if p.requires_grad: penalty += torch.sum(fisher[name] * (p - theta_star[name]) ** 2)
    return 0.5 * ewc_lambda * penalty

def online_ewc_penalty(model, online_fisher, online_theta_star, ewc_lambda):
    if not online_fisher or not online_theta_star or ewc_lambda <= 0: return torch.tensor(0.0, device=cfg.device)
    penalty = torch.tensor(0.0, device=cfg.device)
    for name, p in model.named_parameters():
        if p.requires_grad: penalty += torch.sum(online_fisher[name] * (p - online_theta_star[name]) ** 2)
    return 0.5 * ewc_lambda * penalty

def init_si_state(model): return {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}, copy_params(model)

def si_penalty(model, si_importance, si_prev_params, si_c):
    if si_c <= 0: return torch.tensor(0.0, device=cfg.device)
    penalty = torch.tensor(0.0, device=cfg.device)
    for name, p in model.named_parameters():
        if p.requires_grad: penalty += torch.sum(si_importance[name] * (p - si_prev_params[name]) ** 2)
    return si_c * penalty

def update_memory(memory, loader, memory_budget, seed):
    if memory_budget <= 0: return {"x": [], "y": [], "task_id": []}
    xs, ys, tids = list(memory["x"]), list(memory["y"]), list(memory["task_id"])
    for x, y, task_ids in loader:
        for i in range(x.size(0)):
            xs.append(x[i].cpu()), ys.append(y[i].cpu()), tids.append(task_ids[i].cpu())
    groups = defaultdict(list)
    for i, (y, tid) in enumerate(zip(ys, tids)): groups[(int(tid.item()), int(y.item()))].append(i)
    rng = np.random.default_rng(seed)
    keys = sorted(groups.keys())
    if not keys: return {"x": [], "y": [], "task_id": []}
    base_quota = max(1, memory_budget // len(keys))
    remainder = memory_budget % len(keys)
    selected = []
    for k_i, key in enumerate(keys):
        candidates = np.array(groups[key])
        rng.shuffle(candidates)
        quota = base_quota + (1 if k_i < remainder else 0)
        selected.extend(candidates[:min(quota, len(candidates))].tolist())
    if len(selected) < memory_budget:
        rest = [i for i in range(len(xs)) if i not in set(selected)]
        rng.shuffle(rest)
        selected.extend(rest[:memory_budget - len(selected)])
    selected = selected[:memory_budget]
    return {"x": [xs[i] for i in selected], "y": [ys[i] for i in selected], "task_id": [tids[i] for i in selected]}

def sample_memory_batch(memory, batch_size):
    n = len(memory["x"])
    if n == 0: return None
    idx = np.random.choice(n, size=min(batch_size, n), replace=n < batch_size)
    return torch.stack([memory["x"][i] for i in idx]).to(cfg.device), torch.stack([memory["y"][i] for i in idx]).to(cfg.device), torch.stack([memory["task_id"][i] for i in idx]).to(cfg.device)

@torch.no_grad()
def evaluate_ncm(model, memory, test_loaders, protocol, scenario):
    if not memory["x"]: return [0.0] * len(test_loaders)
    model.eval()
    features_by_key = defaultdict(list)
    for x, y, tid in zip(memory["x"], memory["y"], memory["task_id"]):
        _, feat = model(x.unsqueeze(0).to(cfg.device), return_features=True)
        key = (int(tid.item()), int(y.item())) if scenario == "Task-IL" else int(y.item())
        features_by_key[key].append(feat.squeeze(0).cpu())
    means = {k: torch.stack(v).mean(dim=0) for k, v in features_by_key.items()}
    if not means: return [0.0] * len(test_loaders)
    
    accs = []
    for loader in test_loaders:
        correct, total = 0, 0
        for x, y, task_ids in loader:
            _, feats = model(x.to(cfg.device), return_features=True)
            feats, y, task_ids = feats.cpu(), y.cpu(), task_ids.cpu()
            for i in range(feats.size(0)):
                if scenario == "Task-IL":
                    tid_int = int(task_ids[i].item())
                    c_keys = [k for k in means if isinstance(k, tuple) and k[0] == tid_int]
                    pred = c_keys[int(np.argmin([torch.norm(feats[i] - means[k]).item() for k in c_keys]))][1] if c_keys else -999
                else:
                    c_keys = list(means.keys())
                    pred = c_keys[int(np.argmin([torch.norm(feats[i] - means[k]).item() for k in c_keys]))]
                correct += int(pred == int(y[i].item()))
                total += 1
        accs.append(correct / max(total, 1))
    return accs

# ============================================================
# 6. TRAINING SEQUENCE
# ============================================================
def train_sequence(protocol, scenario, method, hyper_value=0.0, memory_budget=0, xdg_percent=0, online_gamma=1.0):
    set_seed(cfg.seed)
    train_loaders, test_loaders = build_loaders(protocol, scenario)
    model = make_model(protocol, scenario)
    optimizer = make_optimizer(model)

    ewc_memory, online_fisher, online_theta_star = [], None, None
    si_importance, si_prev_params = init_si_state(model)
    replay_memory = {"x": [], "y": [], "task_id": []}
    xdg_masks = make_xdg_masks(len(train_loaders), cfg.hidden_sizes, xdg_percent, cfg.seed+999) if method == "XdG" else None

    history_rows = []
    global_epoch = 0

    for task_id, train_loader in enumerate(train_loaders):
        task_start_params = copy_params(model)
        small_omega = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad} if method == "SI" else None

        for epoch in range(cfg.epochs_per_task):
            model.train()
            for x, y, task_ids in train_loader:
                x, y, task_ids = x.to(cfg.device), y.to(cfg.device), task_ids.to(cfg.device)
                optimizer.zero_grad()
                loss = classification_loss(model, x, y, task_ids, protocol, scenario, xdg_masks)
                if method == "EWC": loss += ewc_penalty(model, ewc_memory, float(hyper_value))
                elif method == "OnlineEWC": loss += online_ewc_penalty(model, online_fisher, online_theta_star, float(hyper_value))
                elif method == "SI": loss += si_penalty(model, si_importance, si_prev_params, float(hyper_value))
                elif method in ["Replay", "Replay+NCM", "iCaRL-lite"]:
                    mem_batch = sample_memory_batch(replay_memory, cfg.batch_size)
                    if mem_batch: loss = 0.5 * loss + 0.5 * classification_loss(model, mem_batch[0], mem_batch[1], mem_batch[2], protocol, scenario)

                params_before = copy_params(model) if method == "SI" else None
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                if method == "SI":
                    for name, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            small_omega[name] += -p.grad.detach() * (p.detach() - params_before[name])
            
            global_epoch += 1
            accs = evaluate_all_tasks(model, test_loaders, protocol, scenario, xdg_masks)
            row = {"protocol": protocol, "scenario": scenario, "method": method, "hyper_value": hyper_value,
                   "memory_budget": memory_budget, "global_epoch": global_epoch,
                   "average_accuracy": float(np.mean(accs))}
            history_rows.append(row)

        if method == "EWC":
            ewc_memory.append({"theta_star": copy_params(model), "fisher": compute_fisher(model, train_loader, protocol, scenario, xdg_masks)})
        elif method == "OnlineEWC":
            fisher = compute_fisher(model, train_loader, protocol, scenario, xdg_masks)
            online_fisher = {n: online_gamma * online_fisher[n] + fisher[n] for n in fisher} if online_fisher else fisher
            online_theta_star = copy_params(model)
        elif method == "SI":
            params_after = copy_params(model)
            for n in si_importance: si_importance[n] += torch.clamp(small_omega[n] / ((params_after[n] - task_start_params[n])**2 + cfg.si_xi), min=0.0)
            si_prev_params = copy_params(model)
        elif method in ["Replay", "Replay+NCM", "NCM", "iCaRL-lite"]:
            replay_memory = update_memory(replay_memory, train_loader, memory_budget, cfg.seed + 3000 + task_id)

    final_avg_ncm = float(np.mean(evaluate_ncm(model, replay_memory, test_loaders, protocol, scenario))) if method in ["NCM", "Replay+NCM", "iCaRL-lite"] else np.nan
    final_avg_network = float(np.mean(evaluate_all_tasks(model, test_loaders, protocol, scenario, xdg_masks)))
    return {"protocol": protocol, "scenario": scenario, "method": method, "hyper_value": hyper_value, "memory_budget": memory_budget,
            "average_accuracy": final_avg_ncm if method in ["NCM", "Replay+NCM", "iCaRL-lite"] else final_avg_network,
            "history": pd.DataFrame(history_rows)}

# ============================================================
# 7. PAPER-STYLE PLOTTING
# ============================================================
def set_paper_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_figD_gridsearch(df, protocol, filename):
    scenarios = ["Task-IL", "Domain-IL", "Class-IL"]
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 8.0))
    plt.subplots_adjust(hspace=0.45, wspace=0.25)
    
    online_colors = ["#ffcc00", "#ff6600", "#d62728"]  # Yellow to Red gradient

    for r, scenario in enumerate(scenarios):
        sub = df[df["scenario"] == scenario]
        none_val = float(sub[sub["method_family"] == "None"]["average_accuracy"].iloc[0])

        for c, metric in enumerate(["EWC", "SI", "XdG"]):
            ax = axes[r, c]
            set_paper_spines(ax)
            if metric == "XdG" and scenario != "Task-IL":
                ax.axis("off")
                continue
            
            ax.axhline(none_val, color="0.55", linewidth=1.5, label="None")
            
            if metric == "EWC":
                ewc = sub[sub["method_family"] == "EWC"].sort_values("hyperparameter_value")
                if len(ewc) > 0: ax.plot(ewc["hyperparameter_value"], ewc["average_accuracy"], color="#17becf", marker="o", markersize=4, lw=1.8, label="EWC")
                online = sub[sub["method_family"] == "Online EWC"]
                for i, gamma in enumerate(sorted(online["online_gamma"].dropna().unique(), reverse=True)):
                    og = online[online["online_gamma"] == gamma].sort_values("hyperparameter_value")
                    ax.plot(og["hyperparameter_value"], og["average_accuracy"], color=online_colors[i % len(online_colors)], marker="o", markersize=4, lw=1.8, label=f"Online EWC γ={gamma:g}")
                ax.set_xscale("log")
                if r == 2: ax.set_xlabel("EWC: lambda (log-scale)")

            elif metric == "SI":
                si = sub[sub["method_family"] == "SI"].sort_values("hyperparameter_value")
                if len(si) > 0: ax.plot(si["hyperparameter_value"], si["average_accuracy"], color="#8cce8c", marker="o", markersize=4, lw=1.8, label="SI")
                ax.set_xscale("log")
                if r == 2: ax.set_xlabel("SI: c (log-scale)")

            elif metric == "XdG":
                xdg = sub[sub["method_family"] == "XdG"].sort_values("hyperparameter_value")
                if len(xdg) > 0: ax.plot(xdg["hyperparameter_value"], xdg["average_accuracy"], color="#800080", marker="o", markersize=4, lw=1.8, label="XdG")
                if r == 0: ax.set_xlabel("XdG: % of nodes gated")

            if c == 1: ax.set_title(scenario, fontsize=11, fontweight="bold")
            if c == 0: ax.set_ylabel("Average accuracy")
            if r == 0 and c == 2: ax.legend(frameon=False, fontsize=8)
            elif r == 2 and c == 0: ax.legend(frameon=False, fontsize=8)

    path = os.path.join(cfg.output_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close()

def plot_figC_memory_budget(df, filename):
    scenarios = ["Task-IL", "Domain-IL", "Class-IL"]
    protocols = ["split", "permuted"]
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 8.2))
    plt.subplots_adjust(hspace=0.45, wspace=0.25)
    
    line_specs = {
        "Replay": {"c": "black", "ls": "-", "m": "o"},
        "NCM": {"c": "0.55", "ls": "-", "m": "o"},
        "Replay+NCM": {"c": "black", "ls": "--", "m": "^"},
        "iCaRL-lite": {"c": "#8c564b", "ls": "-", "m": None}
    }

    for c, protocol in enumerate(protocols):
        for r, scenario in enumerate(scenarios):
            ax = axes[r, c]
            set_paper_spines(ax)
            sub = df[(df["protocol"] == protocol) & (df["scenario"] == scenario)]
            
            if r == 0:
                letter = 'A' if c == 0 else 'B'
                ax.text(-0.2, 1.1, letter, transform=ax.transAxes, fontsize=16, fontweight='bold')
            
            ax.set_title(scenario, fontsize=11, fontweight="bold")
            if c == 0: ax.set_ylabel("Average accuracy")
            if r == 2: ax.set_xlabel("Total memory budget (log-scale)")

            mem_sub = sub[sub["plot_group"] == "memory_budget"]
            budgets = np.array(sorted(mem_sub["memory_budget"].dropna().unique()), dtype=float) if not mem_sub.empty else []
            if len(budgets) > 0:
                ax.set_xscale("log")
                for method, spec in line_specs.items():
                    m = mem_sub[mem_sub["method"] == method].sort_values("memory_budget")
                    if not m.empty:
                        ax.plot(m["memory_budget"], m["average_accuracy"], color=spec["c"], linestyle=spec["ls"], marker=spec["m"], lw=1.8, label=method if r==2 and c==1 else None)
            
            # Draw baselines as horizontal dashed lines like PDF
            other = sub[sub["plot_group"] == "other_methods"]
            for _, row in other.iterrows():
                if row["method"] == "None baseline": ax.axhline(row["average_accuracy"], color="0.75", ls="-", lw=1.2)
                elif row["method"] == "EWC": ax.axhline(row["average_accuracy"], color="#17becf", ls="--", lw=1.2)

    if not axes[2, 1].lines: pass
    axes[2, 1].legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 0.1))
    
    path = os.path.join(cfg.output_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=cfg.dpi)
    plt.close()

# ============================================================
# 8. RUN SIMULATION & ZIP EXPORT
# ============================================================
def run_all():
    named_dfs = {}
    rows_d, rows_c = [], []

    # Run None Baselines
    for prot in ["split", "permuted"]:
        for scen in ["Task-IL", "Domain-IL", "Class-IL"]:
            res = train_sequence(prot, scen, "None")
            rows_d.append({"protocol": prot, "scenario": scen, "method_family": "None", "hyperparameter_value": 1.0, "average_accuracy": res["average_accuracy"]})
            rows_c.append({"protocol": prot, "scenario": scen, "plot_group": "other_methods", "method": "None baseline", "memory_budget": np.nan, "average_accuracy": res["average_accuracy"]})

    # Run EWC, OnlineEWC, SI, XdG (Grid Search)
    for prot in ["split", "permuted"]:
        for scen in ["Task-IL", "Domain-IL", "Class-IL"]:
            for lam in cfg.ewc_lambda_grid:
                res = train_sequence(prot, scen, "EWC", hyper_value=lam)
                rows_d.append({"protocol": prot, "scenario": scen, "method_family": "EWC", "hyperparameter_value": lam if lam>0 else 1.0, "average_accuracy": res["average_accuracy"]})
                for gam in cfg.online_ewc_gamma_grid:
                    res_og = train_sequence(prot, scen, "OnlineEWC", hyper_value=lam, online_gamma=gam)
                    rows_d.append({"protocol": prot, "scenario": scen, "method_family": "Online EWC", "hyperparameter_value": lam if lam>0 else 1.0, "online_gamma": gam, "average_accuracy": res_og["average_accuracy"]})
            
            for cval in cfg.si_c_grid:
                res_si = train_sequence(prot, scen, "SI", hyper_value=cval)
                rows_d.append({"protocol": prot, "scenario": scen, "method_family": "SI", "hyperparameter_value": cval if cval>0 else 0.01, "average_accuracy": res_si["average_accuracy"]})
            
            if scen == "Task-IL":
                for pct in cfg.xdg_percent_grid:
                    res_xdg = train_sequence(prot, scen, "XdG", hyper_value=pct, xdg_percent=int(pct))
                    rows_d.append({"protocol": prot, "scenario": scen, "method_family": "XdG", "hyperparameter_value": pct, "average_accuracy": res_xdg["average_accuracy"]})

    # Run Memory Models
    for prot in ["split", "permuted"]:
        budgets = cfg.split_memory_budgets if prot == "split" else cfg.permuted_memory_budgets
        for scen in ["Task-IL", "Domain-IL", "Class-IL"]:
            for budg in budgets:
                for meth in ["Replay", "NCM", "Replay+NCM", "iCaRL-lite"]:
                    res_mem = train_sequence(prot, scen, meth, memory_budget=budg)
                    rows_c.append({"protocol": prot, "scenario": scen, "plot_group": "memory_budget", "method": meth, "memory_budget": budg, "average_accuracy": res_mem["average_accuracy"]})

    df_d = pd.DataFrame(rows_d)
    df_c = pd.DataFrame(rows_c)
    
    # Save & Plot
    plot_figD_gridsearch(df_d[df_d["protocol"] == "split"], "split", "figD_split_gridsearch_real_scaledown.png")
    plot_figD_gridsearch(df_d[df_d["protocol"] == "permuted"], "permuted", "figD_permuted_gridsearch_real_scaledown.png")
    plot_figC_memory_budget(df_c, "figC_memory_budget_real_scaledown.png")
    
    df_d.to_csv(os.path.join(cfg.output_dir, "grid_search_matrix.csv"), index=False)
    df_c.to_csv(os.path.join(cfg.output_dir, "memory_budget_matrix.csv"), index=False)
    
    zip_path = shutil.make_archive(cfg.output_dir, "zip", cfg.output_dir)
    print(f"Simulation completed! Files saved to: {zip_path}")
    
    try:
        from google.colab import files
        files.download(zip_path)
    except ImportError:
        pass

if __name__ == "__main__":
    run_all()