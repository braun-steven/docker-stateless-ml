#!/usr/bin/env python3
import random

import numpy as np
import torch

from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn, RatSpnConfig


def set_seed(seed: int):
    """
    Set the seed globally for python, numpy and torch.

    Args:
        seed (int): Seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(seed,):
    # Set the seed
    set_seed(seed)

    # Setup RatSpn Configuration
    config = RatSpnConfig()
    config.in_features = 4  # You can also use config.F as an alias
    config.R = 1  # Number of repetitions
    config.D = 2  # Tree depth
    config.I = 2  # Number of input distributions
    config.S = 2  # Number of parallel sum nodes per sum layer
    config.C = 1  # Number of classes (C=1 for density estimation)
    config.dropout = 0.0  # Dropout value
    config.leaf_base_class = RatNormal  # Leaf base distribution class
    config.leaf_base_kwargs = {}

    # Construct RatSpn from config
    device = torch.device("cuda:0")
    model = RatSpn(config).to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Train loop
    for i in range(10):
        x = torch.randn(10, config.in_features, device=device)

        # Reset gradients
        optimizer.zero_grad()

        # Inference
        output = model(x)

        # Compute loss
        loss = -output.mean()

        # Backprop
        loss.backward()
        optimizer.step()

    # Evaluate on a random tensor
    x = torch.randn(1, config.in_features, device=device)
    x[0, : config.in_features // 2] = float("nan")
    ll = model(x).mean()
    ll.backward()
    optimizer.step()
    print("--- Model ---")
    print(model)
    print()
    print("--- Likelihood ---")
    print(ll)
    print()
    print("--- Device ---")
    print(ll.device)


if __name__ == "__main__":
    run(0)
