import json
import os
import argparse
import sys
from datetime import datetime
import numpy as np

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(JSONEncoder, self).default(obj)

def save_summary(experiment_dir, metrics, args):
    """Saves experiment summary including metrics and arguments."""
    os.makedirs(experiment_dir, exist_ok=True)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args) if hasattr(args, '__dict__') else args,
        "metrics": metrics
    }
    
    file_path = os.path.join(experiment_dir, "summary.json")
    with open(file_path, 'w') as f:
        json.dump(summary, f, indent=4, cls=JSONEncoder)
    print(f"[INFO] Summary saved to {file_path}")

def get_base_args():
    """Concise argument parser setup."""
    parser = argparse.ArgumentParser(description="ELB Experiment")
    parser.add_argument("--out_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser

def set_seed(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
