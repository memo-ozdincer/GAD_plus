"""Pure parser tests for the external g-xTB adapter."""
from __future__ import annotations

from gadplus.calculator.gxtb import _read_gradient, _read_hessian


def test_hessian_dimension_header_is_not_matrix_element(tmp_path):
    dim = 3
    matrix = [float(i) for i in range(dim * dim)]
    path = tmp_path / "hessian"
    path.write_text("$hessian\n3\n" + " ".join(map(str, matrix)) + "\n$end\n")

    hessian = _read_hessian(path, n_atoms=1)
    assert hessian.shape == (3, 3)
    assert hessian[0, 0].item() == 0.0
    assert hessian[-1, -1].item() == 8.0


def test_gradient_uses_final_cartesian_block(tmp_path):
    path = tmp_path / "gradient"
    path.write_text("$grad\n2 1.0 2.0\n0.1 0.2 0.3\n$end\n")

    gradient = _read_gradient(path, n_atoms=1)
    assert gradient.shape == (1, 3)
    assert gradient.tolist() == [[0.1, 0.2, 0.3]]
