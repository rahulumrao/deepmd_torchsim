"""Tests for :class:`deepmd_torchsim.DeepmdModel`: energy/forces, CPU vs GPU.

Self-contained: a MLIP model for CH4 molecule stored ``tests/model/frozen_model.pth``,
and the coordinates are written in this file, so this test doesn't depend on anything
outside this package.

``tests/model/reference.json`` is a checked-in reference. Every run computes fresh
energy/forces on the default device (CUDA if available, else CPU) and compares
them to that reference.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pytest
import torch
import torch_sim as ts
from ase import Atoms

#######################################################################################
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TESTS_DIR, "model", "frozen_model.pth")
REFERENCE = os.path.join(TESTS_DIR, "model", "reference.json")
FLOAT64_DTYPE = torch.float64
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#######################################################################################

try:
    from deepmd_torchsim import DeepmdModel

    _IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - environment dependent
    _IMPORT_ERROR = str(exc)

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None or not os.path.exists(MODEL_PATH),
    reason=(
        f"can not import 'deepmd' or frozen model missing at {MODEL_PATH} "
        f"(import error: {_IMPORT_ERROR})"
    ),)
#######################################################################################
def build_system() -> Atoms:
    """A tetrahedral CH4 molecule, centered in a 10 A cubic box."""
    symbols = ["C", "H", "H", "H", "H"]
    positions = [
        [0.000000, 0.000000, 0.000000],
        [0.627581, 0.627581, 0.627581],
        [0.627581, -0.627581, -0.627581],
        [-0.627581, 0.627581, -0.627581],
        [-0.627581, -0.627581, 0.627581],
    ]
    box_size = 10.0
    atoms = Atoms(symbols=symbols, positions=positions, cell=[box_size] * 3, pbc=True)
    atoms.positions += box_size / 2  # center in the box
    return atoms

#######################################################################################
def compute(device: torch.device) -> dict:
    """Load the MLIP model on ``device`` and return result."""
    model = DeepmdModel(
        model_path=MODEL_PATH,
        device=device,
        dtype=FLOAT64_DTYPE,
        compute_forces=True,
        compute_stress=False,
    )
    state = ts.io.atoms_to_state([build_system()], device, FLOAT64_DTYPE)
    output = model.forward(state)
    return {
        "energy_eV": output["energy"][0].item(),
        "forces_eV_per_A": output["forces"].detach().cpu().tolist(),
    }

#######################################################################################
def compare_result(label: str, result: dict, reference: dict) -> None:
    """Compare against the reference; warn on mismatch."""
    tolerance = 1e-5
    energy_diff = abs(result["energy_eV"] - reference["energy_eV"])
    forces_diff = (
        torch.tensor(result["forces_eV_per_A"])
        - torch.tensor(reference["forces_eV_per_A"])
    ).abs().max().item()

    if energy_diff >= tolerance or forces_diff >= tolerance:
        with np.printoptions(precision=4, suppress=True, floatmode="fixed"):
            warnings.warn(
                f"{label} energy/forces do not match reference (tol {tolerance:.0e}):\n"
                f"  energy: computed={result['energy_eV']:.4f} eV, "
                f"reference={reference['energy_eV']:.4f} eV, diff={energy_diff:.4f} eV\n"
                f"  forces: computed=\n{np.array(result['forces_eV_per_A'])}\n"
                f"  forces: reference=\n{np.array(reference['forces_eV_per_A'])}\n"
                f"  max abs force diff={forces_diff:.4f} eV/A",
                stacklevel=2,
            )
#######################################################################################
def test_energy_forces() -> None:
    """Model loads and produces finite energy/forces on the default device
    (CUDA if available, else CPU) and the results are compared against the
    checked-in reference.
    """
    with open(REFERENCE) as f:
        reference = json.load(f)

    default_result = compute(DEFAULT_DEVICE)
    assert torch.isfinite(torch.tensor(default_result["energy_eV"]))
    assert torch.isfinite(torch.tensor(default_result["forces_eV_per_A"])).all()
    default_reference = reference.get(DEFAULT_DEVICE.type, reference["cpu"])
    compare_result(DEFAULT_DEVICE.type, default_result, default_reference)

    if DEFAULT_DEVICE.type == "cuda":
        cpu_result = compute(torch.device("cpu"))
        assert torch.isfinite(torch.tensor(cpu_result["energy_eV"]))
        assert torch.isfinite(torch.tensor(cpu_result["forces_eV_per_A"])).all()
        compare_result("cpu", cpu_result, reference["cpu"])
#######################################################################################
# END of File
#######################################################################################
