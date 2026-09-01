"""DeePMD-kit (PyTorch backend) model wrapper for torch-sim.

Provides a :class:`~torch_sim.models.interface.ModelInterface` implementation for running a DeePMD-kit PyTorch-backend model (`frozen_model.pth`) within torch-sim. The wrapper follows the same packaging and integration conventions as torch-sim's other external model implementations.

Example::

    from deepmd_torchsim import DeepmdModel

    model = DeepmdModel(model_path="frozen_model.pth", device="cuda")
    results = model(sim_state)
    energy = results["energy"]  # [n_systems]
    forces = results["forces"]  # [n_atoms, 3]
    stress = results["stress"]  # [n_systems, 3, 3]

References:
    - DeePMD-kit: https://github.com/deepmodeling/deepmd-kit
    - torch-sim ModelInterface: torch_sim/models/interface.py
"""
#######################################################################################
from __future__ import annotations

import traceback
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from ase.data import atomic_numbers as ase_atomic_numbers
from torch_sim.models.interface import ModelInterface

if TYPE_CHECKING:
    from torch_sim.state import SimState

# Importing this module must not fail if `deepmd` isn't installed; the
# ImportError is deferred until someone constructs a DeepmdModel.
try:
    from deepmd.infer import DeepPot

    _IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    warnings.warn(f"deepmd import failed: {traceback.format_exc()}", stacklevel=2)
    _IMPORT_ERROR = exc
#######################################################################################

class DeepmdModel(ModelInterface):
    """torch-sim wrapper for a frozen DeePMD-kit PyTorch model.

    Loads ``frozen_model.pth`` using ``deepmd.infer.DeepPot`` and evaluates
    :class:`~torch_sim.state.SimState` objects, returning batched energies,
    forces, and stresses.

    Forward evaluation:
        Systems are grouped by atom count and species ordering. Since
        ``DeepPot.eval`` can evaluate multiple frames in a single call only
        when they share the same ``atom_types`` array, each group is evaluated
        independently and the results are scattered back into the original
        batch order.

        Groups containing a single system require only one ``eval`` call, so
        mixed-size batches have the same evaluation cost as an ungrouped
        implementation. Grouping improves performance when batches contain
        repeated systems, such as parallel replicas.

    Species mapping:
        DeePMD species indices are determined by their position in the model's
        ``type_map`` (for example, ``["O", "H"]``), rather than by atomic
        number. The mapping is read from ``DeepPot.get_type_map()`` when the
        model is initialized and is not hardcoded.

    Stress convention:
        Stress is computed as

        ``stress = -0.5 * (virial + virial.T) / volume``

        corresponding to the Cauchy stress convention with tensile stress
        positive. This matches the conventions used by DeePMD's ASE
        calculator and torch-sim's ``pair_potential.py``.

    Attributes:
        type_map (list[str]): Element symbols in the species-index order
            defined by the frozen DeePMD model.

    Examples:
        ```py
        model = DeepmdModel(
            model_path="frozen_model.pth",
            device=torch.device("cuda"),
            compute_forces=True,
            compute_stress=True,
        )
        results = model(sim_state)
        ```
"""

    def __init__(
        self,
        model_path: str | Path,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
        *,
        compute_forces: bool = True,
        compute_stress: bool = True,
        head: str | None = None,
    ) -> None:
        """Initialize the DeePMD-kit model wrapper.

        Args:
            model_path: Path to a frozen DeePMD-kit PyTorch-backend model
                (``frozen_model.pth``).
            device: Device the *output tensors* are placed on. Defaults to ``CUDA``
                if available, else CPU. Note that ``DeepPot`` itself manages its
                own internal device placement (it uses CUDA automatically if
                available, independent of this argument); this argument only
                controls where the torch tensors returned by :meth:`forward` live.
            dtype: Floating-point dtype for the returned tensors. Defaults to
                ``torch.float64`` to match the training precision of the example
                se_e2_a water model this package was validated against.
            compute_forces: Whether to compute and return atomic forces.
                Defaults to True.
            compute_stress: Whether to compute and return the stress tensor.
                Defaults to True.
            head: Task/domain head to select, for multitask models such as
                DPA-3 foundation checkpoints (e.g. ``"Omat24"``). Ignored by
                single-task models. Defaults to None, which lets ``DeepPot``
                fall back to a model's own "Default" head if it has one, or
                raise if the model is multitask and ambiguous.
        """
        if _IMPORT_ERROR is not None:
            raise _IMPORT_ERROR
        super().__init__()
        self._device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._memory_scales_with = "n_atoms"

        self.model_path = Path(model_path)
        self._dp = DeepPot(str(self.model_path.resolve()), head=head)

        # Read the type_map from the frozen model's own metadata rather than
        # hardcoding an atomic-number -> DeePMD-type-index mapping.
        self.type_map: list[str] = self._dp.get_type_map()
        self._atomic_number_to_type_index: dict[int, int] = {
            ase_atomic_numbers[symbol]: type_idx
            for type_idx, symbol in enumerate(self.type_map)
        }

    def _atom_types_for_system(self, atomic_numbers: torch.Tensor) -> np.ndarray:
        """Map a system's atomic numbers to DeePMD type-map indices.

        Args:
            atomic_numbers: Atomic numbers for the atoms in one system, shape
                ``[n_atoms_in_system]``.

        Returns:
            np.ndarray: DeePMD type indices, shape ``[n_atoms_in_system]``, dtype
                int.

        Raises:
            ValueError: If an atomic number is not present in the frozen model's
                ``type_map`` (i.e. the model was not trained on that element).
        """
        numbers = atomic_numbers.detach().cpu().numpy().tolist()
        try:
            return np.array(
                [self._atomic_number_to_type_index[z] for z in numbers], dtype=int
            )
        except KeyError as exc:
            missing_z = exc.args[0]
            raise ValueError(
                f"Atomic number {missing_z} is not in the frozen model's type_map "
                f"{self.type_map}; this model cannot evaluate that element."
            ) from exc

    def _eval_group(
        self,
        positions_list: list[np.ndarray],
        cell_list: list[np.ndarray],
        atom_types: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a group of same-shape, same-species-order systems in one call.

        All systems in the group share one ``atom_types`` array (same atom
        count and species order) — see :meth:`forward` for how groups are
        built. A group of size 1 is just a single-frame ``eval`` call.

        Args:
            positions_list: One ``[n_atoms, 3]`` position array (Angstrom) per
                system in the group.
            cell_list: One ``[3, 3]`` cell array per system, torch-sim's
                column-vector convention (see :class:`~torch_sim.state.SimState`).
            atom_types: DeePMD type indices shared by every system in the
                group, shape ``[n_atoms]``.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: ``(energy, forces,
            virial)`` where ``energy`` has shape ``[n_frames]``, ``forces`` has
            shape ``[n_frames, n_atoms, 3]`` (eV/Angstrom), and ``virial`` has
            shape ``[n_frames, 3, 3]`` (eV).
        """
        n_frames = len(positions_list)
        coords = np.stack(
            [p.astype(np.float64).reshape(-1) for p in positions_list], axis=0
        )

        # torch-sim stores cell column-vector-wise: [[a1,b1,c1],[a2,b2,c2],[a3,b3,c3]].
        # DeepPot / ASE expect the row-vector convention
        # [[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]], i.e. the transpose.
        cells = np.stack([c.astype(np.float64).T.reshape(-1) for c in cell_list], axis=0)

        # atomic/fparam/aparam/mixed_type are passed explicitly to match one of
        # DeepPot.eval's typed @overload stubs, which don't default them.
        energy, force, virial = self._dp.eval(
            coords=coords,
            cells=cells,
            atom_types=atom_types,
            atomic=False,
            fparam=None,
            aparam=None,
            mixed_type=False,
        )[:3]
        return energy[:, 0], force, virial.reshape(n_frames, 3, 3)

    def forward(self, state: SimState, **_kwargs) -> dict[str, torch.Tensor]:
        """Compute energy, forces, and stress for a (possibly batched) state.

        Groups the systems present in ``state.system_idx`` by identical atom
        count and species order, issues one ``DeepPot.eval`` call per group
        (see :meth:`_eval_group` and the class docstring), and scatters the
        per-system results back into batched output tensors.

        Args:
            state (SimState): Simulation state containing:
                - positions: Atomic positions with shape [n_atoms, 3]
                - cell: Unit cell vectors with shape [n_systems, 3, 3]
                - system_idx: System indices for each atom with shape [n_atoms]
                - atomic_numbers: Atomic numbers with shape [n_atoms]

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - "energy": Potential energy, shape [n_systems] (eV)
                - "forces": Atomic forces, shape [n_atoms, 3] (eV/Angstrom;
                    only if compute_forces=True)
                - "stress": Cauchy stress, shape [n_systems, 3, 3]
                    (eV/Angstrom^3; only if compute_stress=True)
        """
        n_systems = int(state.system_idx.max().item()) + 1
        n_atoms = state.positions.shape[0]

        energies = torch.zeros(n_systems, dtype=self._dtype, device=self._device)
        forces_out = (
            torch.zeros((n_atoms, 3), dtype=self._dtype, device=self._device)
            if self._compute_forces
            else None
        )
        stress_out = (
            torch.zeros((n_systems, 3, 3), dtype=self._dtype, device=self._device)
            if self._compute_stress
            else None
        )

        system_masks = [state.system_idx == sys_idx for sys_idx in range(n_systems)]
        system_atom_types = [
            self._atom_types_for_system(state.atomic_numbers[mask])
            for mask in system_masks
        ]

        # Group systems by atom-type sequence; DeepPot.eval only batches frames
        # that share one atom_types array. Unmatched systems form a group of 1.
        groups: dict[tuple[int, ...], list[int]] = {}
        for sys_idx, atype in enumerate(system_atom_types):
            groups.setdefault(tuple(atype.tolist()), []).append(sys_idx)

        for sys_indices in groups.values():
            shared_atom_types = system_atom_types[sys_indices[0]]
            positions_list = [
                state.positions[system_masks[i]].detach().cpu().numpy()
                for i in sys_indices
            ]
            cell_list = [state.cell[i].detach().cpu().numpy() for i in sys_indices]

            energy, force, virial = self._eval_group(
                positions_list, cell_list, shared_atom_types
            )

            for local_i, sys_idx in enumerate(sys_indices):
                energies[sys_idx] = float(energy[local_i])
                mask = system_masks[sys_idx]

                if forces_out is not None:
                    forces_out[mask] = torch.tensor(
                        force[local_i], dtype=self._dtype, device=self._device
                    )

                if stress_out is not None:
                    # cell is stored column-vector-wise; volume is basis-independent.
                    volume = torch.abs(torch.det(state.cell[sys_idx])).item()
                    stress = -0.5 * (virial[local_i] + virial[local_i].T) / volume
                    stress_out[sys_idx] = torch.tensor(
                        stress, dtype=self._dtype, device=self._device
                    )

        results: dict[str, torch.Tensor] = {"energy": energies}
        if forces_out is not None:
            results["forces"] = forces_out
        if stress_out is not None:
            results["stress"] = stress_out
        return results
#######################################################################################
# End of File
#######################################################################################