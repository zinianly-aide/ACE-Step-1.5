"""Unit tests for startup model initialization orchestration helper."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

from acestep.api.startup_model_init import do_model_initialization
from acestep.api.startup_model_init import initialize_models_at_startup


def _gpu_config(init_lm_default: bool = True) -> SimpleNamespace:
    """Create a fake GPU config object with all fields used by startup init."""

    return SimpleNamespace(
        gpu_memory_gb=24.0,
        tier="high",
        max_duration_with_lm=180,
        max_duration_without_lm=300,
        max_batch_size_with_lm=2,
        max_batch_size_without_lm=4,
        init_lm_default=init_lm_default,
        available_lm_models=["acestep-5Hz-lm-1.1B"],
    )


class StartupModelInitTests(unittest.TestCase):
    """Behavior tests for startup model-loading orchestration."""

    @patch("acestep.api.startup_model_init.initialize_llm_at_startup")
    @patch("acestep.api.startup_model_init.set_global_gpu_config")
    @patch("acestep.api.startup_model_init.get_gpu_config")
    def test_initialize_models_at_startup_skips_model_init_in_no_init_mode(
        self,
        mock_get_gpu_config: MagicMock,
        _mock_set_global_gpu_config: MagicMock,
        mock_initialize_llm_at_startup: MagicMock,
    ) -> None:
        """Helper should skip model initialization when ACESTEP_NO_INIT resolves true."""

        app = SimpleNamespace(state=SimpleNamespace())
        handler = MagicMock()
        llm_handler = MagicMock()
        mock_get_gpu_config.return_value = _gpu_config()

        def _env_bool(name: str, default: bool) -> bool:
            return True if name == "ACESTEP_NO_INIT" else default

        initialize_models_at_startup(
            app=app,
            handler=handler,
            llm_handler=llm_handler,
            handler2=None,
            handler3=None,
            config_path2="",
            config_path3="",
            get_project_root=MagicMock(return_value="k:/repo"),
            get_model_name=MagicMock(return_value="acestep-v15-turbo"),
            ensure_model_downloaded=MagicMock(),
            env_bool=_env_bool,
        )

        handler.initialize_service.assert_not_called()
        mock_initialize_llm_at_startup.assert_not_called()
        self.assertIsNotNone(getattr(app.state, "gpu_config", None))

    @patch("acestep.api.startup_model_init.initialize_llm_at_startup")
    @patch("acestep.api.startup_model_init.set_global_gpu_config")
    @patch("acestep.api.startup_model_init.get_gpu_config")
    def test_initialize_models_default_lazy_mode_skips_loading(
        self,
        mock_get_gpu_config: MagicMock,
        _mock_set_global_gpu_config: MagicMock,
        mock_initialize_llm_at_startup: MagicMock,
    ) -> None:
        """Default behavior should skip model loading (lazy-load on first request)."""

        app = SimpleNamespace(state=SimpleNamespace())
        handler = MagicMock()
        llm_handler = MagicMock()
        mock_get_gpu_config.return_value = _gpu_config()

        # Use default env_bool — ACESTEP_NO_INIT defaults to True (lazy mode)
        initialize_models_at_startup(
            app=app,
            handler=handler,
            llm_handler=llm_handler,
            handler2=None,
            handler3=None,
            config_path2="",
            config_path3="",
            get_project_root=MagicMock(return_value="k:/repo"),
            get_model_name=MagicMock(return_value="acestep-v15-turbo"),
            ensure_model_downloaded=MagicMock(),
            env_bool=lambda _name, default: default,
        )

        handler.initialize_service.assert_not_called()
        mock_initialize_llm_at_startup.assert_not_called()
        self.assertIsNotNone(getattr(app.state, "gpu_config", None))
        # Init kwargs should be stored for lazy use
        self.assertIsNotNone(getattr(app.state, "_model_init_kwargs", None))

    @patch("acestep.api.startup_model_init.initialize_llm_at_startup")
    @patch("acestep.api.startup_model_init.set_global_gpu_config")
    @patch("acestep.api.startup_model_init.get_gpu_config")
    def test_initialize_models_at_startup_initializes_primary_and_calls_llm(
        self,
        mock_get_gpu_config: MagicMock,
        _mock_set_global_gpu_config: MagicMock,
        mock_initialize_llm_at_startup: MagicMock,
    ) -> None:
        """Helper should initialize primary DiT and then call LLM startup helper
        when ACESTEP_NO_INIT is explicitly set to False."""

        app = SimpleNamespace(
            state=SimpleNamespace(
                _initialized=False,
                _initialized2=False,
                _initialized3=False,
                _init_error=None,
                _llm_initialized=False,
                _llm_init_error=None,
                _llm_lazy_load_disabled=False,
            )
        )
        handler = MagicMock()
        handler.initialize_service.return_value = ("ok", True)
        llm_handler = MagicMock()
        ensure_model_downloaded = MagicMock()
        mock_get_gpu_config.return_value = _gpu_config()

        def _env_bool(name: str, default: bool) -> bool:
            # Force ACESTEP_NO_INIT=False to trigger eager loading
            return False if name == "ACESTEP_NO_INIT" else default

        with patch.dict(os.environ, {}, clear=True):
            initialize_models_at_startup(
                app=app,
                handler=handler,
                llm_handler=llm_handler,
                handler2=None,
                handler3=None,
                config_path2="",
                config_path3="",
                get_project_root=MagicMock(return_value="k:/repo"),
                get_model_name=MagicMock(return_value="acestep-v15-turbo"),
                ensure_model_downloaded=ensure_model_downloaded,
                env_bool=_env_bool,
            )

        handler.initialize_service.assert_called_once()
        ensure_model_downloaded.assert_any_call("acestep-v15-turbo", ANY)
        ensure_model_downloaded.assert_any_call("vae", ANY)
        self.assertTrue(app.state._initialized)
        mock_initialize_llm_at_startup.assert_called_once()

    @patch("acestep.api.startup_model_init.initialize_llm_at_startup")
    @patch("acestep.api.startup_model_init.set_global_gpu_config")
    @patch("acestep.api.startup_model_init.get_gpu_config")
    def test_initialize_models_at_startup_raises_on_primary_init_failure(
        self,
        mock_get_gpu_config: MagicMock,
        _mock_set_global_gpu_config: MagicMock,
        mock_initialize_llm_at_startup: MagicMock,
    ) -> None:
        """Helper should raise and persist init error when primary DiT fails."""

        app = SimpleNamespace(
            state=SimpleNamespace(
                _initialized=False,
                _init_error=None,
                _llm_initialized=False,
                _llm_init_error=None,
                _llm_lazy_load_disabled=False,
            )
        )
        handler = MagicMock()
        handler.initialize_service.return_value = ("boom", False)
        mock_get_gpu_config.return_value = _gpu_config()

        def _env_bool(name: str, default: bool) -> bool:
            # Force ACESTEP_NO_INIT=False to trigger eager loading
            return False if name == "ACESTEP_NO_INIT" else default

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                initialize_models_at_startup(
                    app=app,
                    handler=handler,
                    llm_handler=MagicMock(),
                    handler2=None,
                    handler3=None,
                    config_path2="",
                    config_path3="",
                    get_project_root=MagicMock(return_value="k:/repo"),
                    get_model_name=MagicMock(return_value="acestep-v15-turbo"),
                    ensure_model_downloaded=MagicMock(),
                    env_bool=_env_bool,
                )

        mock_initialize_llm_at_startup.assert_not_called()
        self.assertEqual("boom", app.state._init_error)


class StartupOffloadLogTests(unittest.TestCase):
    """The printed offload state must always match the value actually passed to
    initialize_service (regression: "Auto-enabling CPU offload" was printed on
    <16GB GPUs even when ACESTEP_OFFLOAD_TO_CPU=false pinned the behavior)."""

    def _run(self, env: dict, gpu_memory_gb: float) -> tuple[list[str], MagicMock]:
        captured: list[str] = []
        app = SimpleNamespace(
            state=SimpleNamespace(
                gpu_config=SimpleNamespace(
                    gpu_memory_gb=gpu_memory_gb,
                    init_lm_default=True,
                    tier="tier4",
                    max_duration_with_lm=480,
                    max_duration_without_lm=600,
                    max_batch_size_with_lm=2,
                    max_batch_size_without_lm=4,
                    available_lm_models=["acestep-5Hz-lm-0.6B"],
                ),
            )
        )
        handler = MagicMock()
        handler.initialize_service.return_value = ("ok", True)
        llm_handler = MagicMock()
        llm_handler.initialize.return_value = ("ok", True)

        def _env_bool(name: str, default: bool) -> bool:
            raw = env.get(name)
            if raw is None:
                return default
            return str(raw).lower() in ("1", "true", "yes", "on")

        def _getenv(name: str, default=None):
            return env.get(name, default)

        with patch.dict(os.environ, {}, clear=True):
            with patch("acestep.api.startup_model_init.os.getenv", side_effect=_getenv):
                with patch("builtins.print", side_effect=lambda *a: captured.append(" ".join(str(x) for x in a))):
                    do_model_initialization(
                        app=app,
                        handler=handler,
                        llm_handler=llm_handler,
                        handler2=None,
                        handler3=None,
                        config_path2=None,
                        config_path3=None,
                        get_project_root=MagicMock(return_value="/tmp/proj"),
                        get_model_name=MagicMock(return_value="acestep-v15-turbo"),
                        ensure_model_downloaded=MagicMock(return_value="/tmp/proj/checkpoints"),
                        env_bool=_env_bool,
                    )
        return captured, handler

    def test_env_false_on_small_gpu_disables_and_logs_override(self) -> None:
        """Explicit false must disable offload and log override (never auto)."""
        captured, handler = self._run({"ACESTEP_OFFLOAD_TO_CPU": "false"}, gpu_memory_gb=11.8)
        self.assertFalse(handler.initialize_service.call_args.kwargs["offload_to_cpu"])
        joined = "\n".join(captured)
        self.assertIn("CPU offload: disabled by explicit env override", joined)
        self.assertNotIn("Auto-enabling", joined)
        self.assertNotIn("auto-enabled", joined)

    def test_env_true_on_large_gpu_enables_and_logs_override(self) -> None:
        """Explicit true must enable offload and log override."""
        captured, handler = self._run({"ACESTEP_OFFLOAD_TO_CPU": "true"}, gpu_memory_gb=24.0)
        self.assertTrue(handler.initialize_service.call_args.kwargs["offload_to_cpu"])
        self.assertIn("CPU offload: enabled by explicit env override", "\n".join(captured))

    def test_no_env_small_gpu_keeps_auto_enable(self) -> None:
        """Without env override a <16GB GPU keeps the auto-offload path."""
        captured, handler = self._run({}, gpu_memory_gb=11.8)
        self.assertTrue(handler.initialize_service.call_args.kwargs["offload_to_cpu"])
        self.assertIn("CPU offload: auto-enabled (GPU < 16GB)", "\n".join(captured))

    def test_no_env_large_gpu_disabled_by_detection(self) -> None:
        """Without env override a >=16GB GPU disables offload via detection."""
        captured, handler = self._run({}, gpu_memory_gb=32.0)
        self.assertFalse(handler.initialize_service.call_args.kwargs["offload_to_cpu"])
        self.assertIn("CPU offload: disabled by auto-detection (GPU >= 16GB)", "\n".join(captured))

    def test_no_gpu_runs_on_cpu(self) -> None:
        """Zero GPU memory reports CPU-only mode with offload disabled."""
        captured, handler = self._run({}, gpu_memory_gb=0.0)
        self.assertFalse(handler.initialize_service.call_args.kwargs["offload_to_cpu"])
        self.assertIn("No GPU detected, running on CPU", "\n".join(captured))


if __name__ == "__main__":
    unittest.main()
