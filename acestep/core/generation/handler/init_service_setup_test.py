"""Unit tests for MLX backend initialization gating in InitServiceSetupMixin.

Verifies the ACESTEP_MLX_VAE=0/false path skips ``_init_mlx_vae()`` entirely
(no conversion, no mx.compile pre-compilation, no resident MLX VAE copy),
while the default/1/true path still invokes it (regression for the M4/16GB
profile which uses PyTorch/MPS tiled VAE decode).
"""
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def _load_handler_module(filename: str, module_name: str) -> types.ModuleType:
    """Load a handler mixin module for isolated tests."""
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    package_paths = {
        "acestep": repo_root / "acestep",
        "acestep.core": repo_root / "acestep" / "core",
        "acestep.core.generation": repo_root / "acestep" / "core" / "generation",
        "acestep.core.generation.handler": repo_root / "acestep" / "core" / "generation" / "handler",
    }
    previous_modules = {name: sys.modules.get(name) for name in package_paths}
    try:
        for package_name, package_path in package_paths.items():
            if package_name in sys.modules:
                continue
            package_module = types.ModuleType(package_name)
            package_module.__path__ = [str(package_path)]
            sys.modules[package_name] = package_module
        module_path = Path(__file__).with_name(filename)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec for {module_name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for package_name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(package_name, None)
            else:
                sys.modules[package_name] = previous


SETUP_MODULE = _load_handler_module(
    "init_service_setup.py",
    "acestep.core.generation.handler.init_service_setup",
)
InitServiceSetupMixin = SETUP_MODULE.InitServiceSetupMixin


class _BackendHost(InitServiceSetupMixin):
    """Host exposing the minimal MLX backend state for gating tests."""

    def __init__(self):
        self.mlx_decoder = object()
        self.use_mlx_dit = True
        self.mlx_vae = object()
        self.use_mlx_vae = True
        self._init_mlx_dit = Mock(return_value=True)
        self._init_mlx_vae = Mock(return_value=True)


class MlXBackendGatingTests(unittest.TestCase):
    """ACESTEP_MLX_VAE controls whether _init_mlx_vae is called."""

    def test_mlx_vae_zero_skips_initialization(self) -> None:
        """ACESTEP_MLX_VAE=0 must not call _init_mlx_vae and must clear state."""
        host = _BackendHost()
        with patch.dict(os.environ, {"ACESTEP_MLX_VAE": "0"}, clear=False):
            dit_status, vae_status = host._initialize_mlx_backends(
                device="mps", use_mlx_dit=False, mlx_compile_requested=False
            )
        host._init_mlx_vae.assert_not_called()
        self.assertIsNone(host.mlx_vae)
        self.assertFalse(host.use_mlx_vae)
        self.assertIn("Disabled by user", vae_status)

    def test_mlx_vae_false_skips_initialization(self) -> None:
        """ACESTEP_MLX_VAE=false must skip initialization like 0."""
        host = _BackendHost()
        with patch.dict(os.environ, {"ACESTEP_MLX_VAE": "false"}, clear=False):
            _dit_status, vae_status = host._initialize_mlx_backends(
                device="mps", use_mlx_dit=False, mlx_compile_requested=False
            )
        host._init_mlx_vae.assert_not_called()
        self.assertIsNone(host.mlx_vae)
        self.assertFalse(host.use_mlx_vae)
        self.assertIn("Disabled by user", vae_status)

    def test_mlx_vae_one_keeps_initialization(self) -> None:
        """ACESTEP_MLX_VAE=1 must still call _init_mlx_vae."""
        host = _BackendHost()
        with patch.dict(os.environ, {"ACESTEP_MLX_VAE": "1"}, clear=False):
            _dit_status, vae_status = host._initialize_mlx_backends(
                device="mps", use_mlx_dit=False, mlx_compile_requested=False
            )
        host._init_mlx_vae.assert_called_once()
        self.assertIn("Active", vae_status)

    def test_mlx_vae_default_keeps_initialization(self) -> None:
        """Unset ACESTEP_MLX_VAE keeps the original default path (initialize)."""
        host = _BackendHost()
        with patch.dict(os.environ, {}, clear=True):
            _dit_status, vae_status = host._initialize_mlx_backends(
                device="mps", use_mlx_dit=False, mlx_compile_requested=False
            )
        host._init_mlx_vae.assert_called_once()
        self.assertIn("Active", vae_status)

    def test_mlx_vae_zero_on_cpu_also_skips(self) -> None:
        """The guard applies on cpu devices too."""
        host = _BackendHost()
        with patch.dict(os.environ, {"ACESTEP_MLX_VAE": "0"}, clear=False):
            _dit_status, vae_status = host._initialize_mlx_backends(
                device="cpu", use_mlx_dit=False, mlx_compile_requested=False
            )
        host._init_mlx_vae.assert_not_called()
        self.assertIsNone(host.mlx_vae)
        self.assertFalse(host.use_mlx_vae)

    def test_mlx_vae_zero_non_apple_device_no_call(self) -> None:
        """Non-mps/cpu devices never initialize MLX VAE regardless of env."""
        host = _BackendHost()
        with patch.dict(os.environ, {"ACESTEP_MLX_VAE": "1"}, clear=False):
            _dit_status, vae_status = host._initialize_mlx_backends(
                device="cuda", use_mlx_dit=False, mlx_compile_requested=False
            )
        host._init_mlx_vae.assert_not_called()
        self.assertIsNone(host.mlx_vae)
        self.assertFalse(host.use_mlx_vae)
        self.assertEqual("Disabled", vae_status)


if __name__ == "__main__":
    unittest.main()
