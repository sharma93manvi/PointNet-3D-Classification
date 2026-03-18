from __future__ import annotations

from typing import Tuple

from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints


def get_modelnet40_datasets(
    root: str,
    num_points: int = 1024,
    force_reload: bool = False,
) -> Tuple[ModelNet, ModelNet]:
    """
    Download + preprocess ModelNet40.

    ModelNet provides meshes (vertices + faces). We use SamplePoints to convert
    each mesh into a fixed-size point cloud (with `data.pos` holding [N, 3]).
    """
    pre_transform = SamplePoints(num=num_points, remove_faces=True)

    train_dataset = ModelNet(
        root=root,
        name="40",
        train=True,
        pre_transform=pre_transform,
        force_reload=force_reload,
    )
    test_dataset = ModelNet(
        root=root,
        name="40",
        train=False,
        pre_transform=pre_transform,
        force_reload=force_reload,
    )
    return train_dataset, test_dataset

