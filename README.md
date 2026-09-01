# deepmd-torchsim

A [torch-sim](https://github.com/TorchSim/torch-sim) `ModelInterface` implementation
for [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit)'s PyTorch backend.

## Install

The package (`DeepmdModel`, torch-sim `ModelInterface` wrapper) can be installed with either `pip` and `uv`:

```bash
# from PyPI (once published) or a local checkout
pip install deepmd-torchsim
uv pip install deepmd-torchsim
uv add deepmd-torchsim

# editable, from a local checkout (what this repo uses during development)
pip install -e .
uv pip install -e .
```

### Getting a working `deepmd-kit` backend

```bash
pip install "deepmd-torchsim[deepmd]"
uv pip install "deepmd-torchsim[deepmd]"
```

The `deepmd` extra pins an working `deepmd-kit==3.1.3`, with `torch==2.10.0`.

## Usage

```python
import torch
from deepmd_torchsim import DeepmdModel
import torch_sim as ts
from ase.build import molecule

model = DeepmdModel(
    model_path="frozen_model.pth",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    compute_forces=True,
    compute_stress=True,
)

state = ts.io.atoms_to_state([molecule("H2O")], model.device, model.dtype)
results = model(state)
print(results["energy"])  # [n_systems]
print(results["forces"])  # [n_atoms, 3]
print(results["stress"])  # [n_systems, 3, 3]
```

For multitask/multi-domain foundation checkpoints (e.g. DPA-3), pass `head=`
to select which trained domain to evaluate with:

```python
model = DeepmdModel(model_path="DPA-3.1-3M.pt", head="Omat24")
```
## Tests

```bash
pytest tests/
```

## License

This project is licensed under the [MIT License](./LICENSE).

## Author
Rahul Verma \
Email: rverma7@ncsu.edu

