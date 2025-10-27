import importlib

import pytest

pytest.importorskip("torch")


def test_import_root_package():
    module = importlib.import_module("lcpfn")
    assert hasattr(module, "predict_curve")
    assert hasattr(module, "load_pretrained")
