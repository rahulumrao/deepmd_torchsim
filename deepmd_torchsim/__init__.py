"""DeePMD-kit (PyTorch backend) integration for torch-sim.

https://github.com/TorchSim/torch-sim

Exposes :class:`DeepmdModel`, a :class:`torch_sim.models.interface.ModelInterface`
implementation that wraps a frozen DeePMD-kit PyTorch-backend model
(``frozen_model.pth``, loaded through ``deepmd.infer.DeepPot``).
"""

from importlib import resources

from deepmd_torchsim.model import DeepmdModel

__all__ = ["DeepmdModel", "example_methane_model_path"]


def example_methane_model_path() -> str:
    """Path to the small CH4 model shipped with this package for examples/tutorials.

    Returns:
        str: Path to ``data/methane_frozen_model.pth``.
    """
    return str(resources.files("deepmd_torchsim") / "data" / "methane_frozen_model.pth")
