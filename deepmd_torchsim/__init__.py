"""DeePMD-kit (PyTorch backend) integration for torch-sim.

https://github.com/TorchSim/torch-sim

Exposes :class:`DeepmdModel`, a :class:`torch_sim.models.interface.ModelInterface`
implementation that wraps a frozen DeePMD-kit PyTorch-backend model
(``frozen_model.pth``, loaded through ``deepmd.infer.DeepPot``).
"""

from deepmd_torchsim.model import DeepmdModel

__all__ = ["DeepmdModel"]
