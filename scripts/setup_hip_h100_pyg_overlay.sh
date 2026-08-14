#!/bin/bash
# Build the small HIP-compatible PyG overlay without mutating the shared venv.
set -euo pipefail

TARGET=${HIP_PYG_OVERLAY:-/scratch/memoozd/gadplus/envs/hip-pyg27-overlay/lib/python3.11/site-packages}
WORK_ROOT=/scratch/memoozd/GAD/GAD_plus

mkdir -p "$TARGET"
cd "$WORK_ROOT"
uv pip install --python .venv/bin/python --target "$TARGET" --no-deps \
  'torch-geometric==2.7.0'
uv pip install --python .venv/bin/python --target "$TARGET" --no-deps \
  'torch-cluster==1.6.3+pt27cu126' \
  -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
uv pip install --python .venv/bin/python --target "$TARGET" --no-deps \
  'torch-scatter==2.1.2+pt27cu126' \
  -f https://data.pyg.org/whl/torch-2.7.0+cu126.html

PYTHONPATH="$TARGET:$WORK_ROOT/src" .venv/bin/python - <<'PY'
import torch
import torch_cluster
import torch_geometric
import torch_scatter
from torch_scatter import segment_coo
from torch_geometric.nn import radius_graph

x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
edges = radius_graph(x, r=1.5, loop=False)
assert edges.shape == (2, 2)
segment = segment_coo(
    torch.tensor([1.0, 2.0, 3.0]),
    torch.tensor([0, 0, 1]),
)
assert torch.equal(segment, torch.tensor([3.0, 3.0]))
print(f"torch={torch.__version__}")
print(f"torch-geometric={torch_geometric.__version__}")
print(f"torch-cluster={torch_cluster.__version__}")
print(f"torch-scatter={torch_scatter.__version__}")
print(f"radius_graph_edges={edges.tolist()}")
print(f"segment_coo={segment.tolist()}")
PY
