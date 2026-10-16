"""Wrong on purpose: a shape bug that only `torchtyc run` can see.

Nothing here is annotated, so `torchtyc check` alone reports nothing.
"""

import torch


def project(x):
    w = torch.empty(16, 8)
    return x @ w  # (4, 8) @ (16, 8): the inner sizes differ


def main():
    project(torch.empty(4, 8))


if __name__ == "__main__":
    main()
