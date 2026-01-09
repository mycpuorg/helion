from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from helion.autotuner.openevolve_tuner import OpenEvolveTuner


# Module-level pickleable objective functions for testing
def _simple_objective(config):
    return 1.0


def _block_size_objective(config):
    return config["block_size"] / 100.0


def _combined_objective(config):
    return config["block_size"] + config["num_warps"]


def _complex_objective(config):
    return config["block_size"] * config["num_warps"] / config["num_stages"]


class TestOpenEvolveTunerInit:
    """Tests for OpenEvolveTuner initialization and validation."""

    def test_init_with_valid_config_space(self):
        config_space = {
            "block_size": [32, 64, 128],
            "num_warps": [1, 2, 4],
        }

        tuner = OpenEvolveTuner(config_space, _simple_objective)

        assert tuner.config_space == config_space
        assert tuner.objective == _simple_objective
        assert tuner.max_evaluations == 100
        assert tuner.population_size == 20
        assert tuner.temperature == 0.2
        assert tuner.verbose is True
        assert tuner.best_config is None
        assert tuner.best_score is None
        assert tuner.evaluation_count == 0
        assert tuner.history == []

    def test_init_with_custom_parameters(self):
        config_space = {"block_size": [32, 64]}

        tuner = OpenEvolveTuner(
            config_space,
            _simple_objective,
            max_evaluations=50,
            population_size=10,
            temperature=0.5,
            model="gpt-4",
            api_base="https://custom.api.com/v1",
            verbose=False,
            checkpoint_interval=5,
            checkpoint_path="/tmp/checkpoints",
            artifact_dir="/tmp/artifacts",
        )

        assert tuner.max_evaluations == 50
        assert tuner.population_size == 10
        assert tuner.temperature == 0.5
        assert tuner.model == "gpt-4"
        assert tuner.api_base == "https://custom.api.com/v1"
        assert tuner.verbose is False
        assert tuner.checkpoint_interval == 5
        assert tuner.checkpoint_path == "/tmp/checkpoints"
        assert tuner.artifact_dir == "/tmp/artifacts"

    def test_init_with_initial_config(self):
        config_space = {"block_size": [32, 64, 128], "num_warps": [1, 2, 4]}
        initial_config = {"block_size": 64, "num_warps": 2}

        tuner = OpenEvolveTuner(
            config_space, _simple_objective, initial_config=initial_config
        )
        assert tuner.initial_config == initial_config

    def test_init_with_llm_models(self):
        config_space = {"block_size": [32, 64]}
        llm_models = [
            {"name": "gpt-4", "weight": 0.6},
            {"name": "claude-3", "weight": 0.4, "temperature": 0.3},
        ]

        tuner = OpenEvolveTuner(
            config_space, _simple_objective, llm_models=llm_models
        )
        assert tuner.llm_models == llm_models

    def test_validate_config_space_empty_raises(self):
        with pytest.raises(ValueError, match="config_space cannot be empty"):
            OpenEvolveTuner({}, _simple_objective)

    def test_validate_config_space_non_list_values_raises(self):
        with pytest.raises(
            ValueError, match="config_space\\['block_size'\\] must be a list"
        ):
            OpenEvolveTuner({"block_size": 32}, _simple_objective)

    def test_validate_config_space_empty_list_raises(self):
        with pytest.raises(
            ValueError, match="config_space\\['block_size'\\] cannot be empty"
        ):
            OpenEvolveTuner({"block_size": []}, _simple_objective)


class TestOpenEvolveTunerGenerateInitialProgram:
    """Tests for the _generate_initial_program method."""

    def test_generate_initial_program_basic(self):
        config_space = {
            "block_size": [32, 64, 128],
            "num_warps": [1, 2, 4],
        }
        tuner = OpenEvolveTuner(config_space, _simple_objective)

        program = tuner._generate_initial_program()

        # Should contain function definition
        assert "def get_kernel_config():" in program
        assert "config = {" in program
        assert "return config" in program

        # Should use first values as defaults
        assert "'block_size': 32" in program
        assert "'num_warps': 1" in program

        # Should include valid values as comments
        assert "Valid values for block_size: [32, 64, 128]" in program
        assert "Valid values for num_warps: [1, 2, 4]" in program

    def test_generate_initial_program_with_initial_config(self):
        config_space = {
            "block_size": [32, 64, 128],
            "num_warps": [1, 2, 4],
        }
        initial_config = {"block_size": 64, "num_warps": 2}
        tuner = OpenEvolveTuner(
            config_space, _simple_objective, initial_config=initial_config
        )

        program = tuner._generate_initial_program()

        # Should use initial_config values
        assert "'block_size': 64" in program
        assert "'num_warps': 2" in program

    def test_generate_initial_program_with_invalid_initial_config(self):
        """If initial_config value is not in valid values, use first valid value."""
        config_space = {"block_size": [32, 64, 128]}
        initial_config = {"block_size": 256}  # Not in valid values
        tuner = OpenEvolveTuner(
            config_space, _simple_objective, initial_config=initial_config
        )

        program = tuner._generate_initial_program()

        # Should fall back to first valid value
        assert "'block_size': 32" in program

    def test_generate_initial_program_string_values(self):
        config_space = {"indexing": ["block_ptr", "tensor_descriptor", "pointer"]}
        tuner = OpenEvolveTuner(config_space, _simple_objective)

        program = tuner._generate_initial_program()

        # Should properly quote string values
        assert "'indexing': 'block_ptr'" in program

    def test_generate_initial_program_is_executable(self):
        config_space = {
            "block_size": [32, 64],
            "num_warps": [2, 4],
            "use_tensor_cores": [True, False],
        }
        tuner = OpenEvolveTuner(config_space, _simple_objective)

        program = tuner._generate_initial_program()

        # Execute the program
        exec_globals: Dict[str, Any] = {}
        exec(program, exec_globals)

        assert "get_kernel_config" in exec_globals
        config = exec_globals["get_kernel_config"]()

        assert config == {"block_size": 32, "num_warps": 2, "use_tensor_cores": True}


class TestOpenEvolveTunerCreateEvaluator:
    """Tests for the _create_evaluator_function method."""

    def test_create_evaluator_writes_files(self):
        config_space = {"block_size": [32, 64]}
        tuner = OpenEvolveTuner(config_space, _block_size_objective)

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator_path = os.path.join(tmpdir, "evaluator.py")
            artifact_dir = os.path.join(tmpdir, "artifacts")

            tuner._create_evaluator_function(evaluator_path, artifact_dir)

            # Check evaluator.py was written
            assert os.path.exists(evaluator_path)

            # Check objective pickle was written
            pickle_path = f"{evaluator_path}.objective.pkl"
            assert os.path.exists(pickle_path)

            # Verify objective can be unpickled
            with open(pickle_path, "rb") as f:
                loaded_objective = pickle.load(f)
            assert loaded_objective({"block_size": 64}) == 0.64

    def test_create_evaluator_contains_config_space(self):
        config_space = {"block_size": [32, 64], "num_warps": [2, 4]}
        tuner = OpenEvolveTuner(config_space, _simple_objective)

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator_path = os.path.join(tmpdir, "evaluator.py")
            tuner._create_evaluator_function(evaluator_path, tmpdir)

            content = Path(evaluator_path).read_text()

            # Should contain config space for validation
            assert "CONFIG_SPACE" in content
            assert "'block_size'" in content
            assert "'num_warps'" in content

    def test_create_evaluator_contains_validation(self):
        config_space = {"block_size": [32, 64]}
        tuner = OpenEvolveTuner(config_space, _simple_objective)

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator_path = os.path.join(tmpdir, "evaluator.py")
            tuner._create_evaluator_function(evaluator_path, tmpdir)

            content = Path(evaluator_path).read_text()

            # Should contain validation function
            assert "def validate_config(config):" in content
            assert "Unknown parameter" in content
            assert "Invalid value" in content
            assert "Missing required parameter" in content


class TestOpenEvolveTunerCreateConfigYaml:
    """Tests for the _create_config_yaml method."""

    def test_create_config_yaml_basic(self):
        config_space = {"block_size": [32, 64]}
        tuner = OpenEvolveTuner(
            config_space,
            _simple_objective,
            max_evaluations=50,
            population_size=15,
            temperature=0.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            tuner._create_config_yaml(config_path)

            assert os.path.exists(config_path)
            content = Path(config_path).read_text()

            assert "max_iterations: 50" in content
            assert "population_size: 15" in content
            assert "temperature: 0.3" in content

    def test_create_config_yaml_contains_system_message(self):
        config_space = {"block_size": [32, 64], "num_warps": [1, 2, 4]}
        tuner = OpenEvolveTuner(config_space, _simple_objective)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            tuner._create_config_yaml(config_path)

            content = Path(config_path).read_text()

            # Should contain system message with allowed values
            assert "block_size: [32, 64]" in content
            assert "num_warps: [1, 2, 4]" in content
            assert "STRICT RULES" in content

    def test_create_config_yaml_with_single_model(self):
        config_space = {"block_size": [32, 64]}
        tuner = OpenEvolveTuner(
            config_space, _simple_objective, model="gpt-4-turbo"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            tuner._create_config_yaml(config_path)

            content = Path(config_path).read_text()
            assert 'name: "gpt-4-turbo"' in content

    def test_create_config_yaml_with_llm_models(self):
        config_space = {"block_size": [32, 64]}
        llm_models = [
            {"name": "gpt-4", "weight": 0.6},
            {"name": "claude-3", "weight": 0.4, "temperature": 0.5},
        ]
        tuner = OpenEvolveTuner(
            config_space, _simple_objective, llm_models=llm_models
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            tuner._create_config_yaml(config_path)

            content = Path(config_path).read_text()

            assert 'name: "gpt-4"' in content
            assert "weight: 0.6" in content
            assert 'name: "claude-3"' in content
            assert "weight: 0.4" in content
            assert "temperature: 0.5" in content


class TestOpenEvolveTunerTune:
    """Tests for the tune() method."""

    def test_tune_raises_import_error_without_openevolve(self):
        config_space = {"block_size": [32, 64]}
        tuner = OpenEvolveTuner(config_space, _simple_objective)

        with patch.dict("sys.modules", {"openevolve": None}):
            with pytest.raises(ImportError, match="OpenEvolve is not installed"):
                tuner.tune()

    def test_tune_with_mocked_openevolve(self):
        config_space = {"block_size": [32, 64], "num_warps": [2, 4]}

        tuner = OpenEvolveTuner(
            config_space, _combined_objective, max_evaluations=5, verbose=False
        )

        # Mock the run_evolution function
        mock_result = MagicMock()
        mock_result.best_code = """
def get_kernel_config():
    config = {
        'block_size': 64,
        'num_warps': 4,
    }
    return config
"""
        mock_result.output_dir = None

        mock_openevolve = MagicMock()
        mock_openevolve.run_evolution = MagicMock(return_value=mock_result)

        with patch.dict("sys.modules", {"openevolve": mock_openevolve}):
            best_config = tuner.tune()

        # Should call run_evolution
        mock_openevolve.run_evolution.assert_called_once()

        # Best config should be extracted from result.best_code
        assert best_config == {"block_size": 64, "num_warps": 4}
        assert tuner.best_config == {"block_size": 64, "num_warps": 4}

    def test_tune_fallback_to_random_search(self):
        """When OpenEvolve doesn't produce valid results, fallback to random search."""
        config_space = {"block_size": [32, 64], "num_warps": [2, 4]}

        tuner = OpenEvolveTuner(
            config_space, _combined_objective, max_evaluations=5, verbose=False
        )

        # Mock run_evolution to return empty result (no best_code)
        mock_result = MagicMock()
        mock_result.best_code = None
        mock_result.output_dir = None

        mock_openevolve = MagicMock()
        mock_openevolve.run_evolution = MagicMock(return_value=mock_result)

        with patch.dict("sys.modules", {"openevolve": mock_openevolve}):
            best_config = tuner.tune()

        # Should still find a valid config via random search fallback
        assert best_config is not None
        assert "block_size" in best_config
        assert "num_warps" in best_config
        assert best_config["block_size"] in [32, 64]
        assert best_config["num_warps"] in [2, 4]

    def test_tune_writes_artifacts(self):
        config_space = {"block_size": [32, 64]}

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = os.path.join(tmpdir, "artifacts")
            tuner = OpenEvolveTuner(
                config_space,
                _block_size_objective,
                max_evaluations=2,
                verbose=False,
                artifact_dir=artifact_dir,
            )

            mock_result = MagicMock()
            mock_result.best_code = """
def get_kernel_config():
    config = {'block_size': 64}
    return config
"""
            mock_result.output_dir = None

            mock_openevolve = MagicMock()
            mock_openevolve.run_evolution = MagicMock(return_value=mock_result)

            with patch.dict("sys.modules", {"openevolve": mock_openevolve}):
                tuner.tune()

            # Artifact directory should be created
            assert os.path.exists(artifact_dir)

            # Summary should be written
            summary_path = os.path.join(artifact_dir, "tuning_summary.json")
            assert os.path.exists(summary_path)

            with open(summary_path) as f:
                summary = json.load(f)
            assert "best_config" in summary
            assert "best_score" in summary


class TestOpenEvolveTunerHistory:
    """Tests for history tracking."""

    def test_history_with_jsonl_file(self):
        config_space = {"block_size": [32, 64]}

        tuner = OpenEvolveTuner(
            config_space, _block_size_objective, max_evaluations=3, verbose=False
        )

        # Create a mock for run_evolution that returns empty result
        mock_result = MagicMock()
        mock_result.best_code = None
        mock_result.output_dir = None

        mock_openevolve = MagicMock()
        mock_openevolve.run_evolution = MagicMock(return_value=mock_result)

        with patch.dict("sys.modules", {"openevolve": mock_openevolve}):
            # This will trigger random search fallback since history won't exist
            best_config = tuner.tune()

        # Best config should exist from fallback
        assert best_config is not None


class TestOpenEvolveTunerIntegration:
    """Integration tests that verify the full flow without actual OpenEvolve."""

    def test_full_setup_without_tune(self):
        """Test that all setup methods work correctly together."""
        config_space = {
            "block_size": [32, 64, 128],
            "num_warps": [2, 4, 8],
            "num_stages": [1, 2, 3],
        }

        initial_config = {"block_size": 64, "num_warps": 4, "num_stages": 2}

        tuner = OpenEvolveTuner(
            config_space,
            _complex_objective,
            max_evaluations=10,
            population_size=5,
            initial_config=initial_config,
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate initial program
            program = tuner._generate_initial_program()
            initial_program_path = os.path.join(tmpdir, "initial_program.py")
            Path(initial_program_path).write_text(program)

            # Execute initial program to verify it works
            exec_globals: Dict[str, Any] = {}
            exec(program, exec_globals)
            config = exec_globals["get_kernel_config"]()
            assert config == initial_config

            # Create evaluator
            evaluator_path = os.path.join(tmpdir, "evaluator.py")
            tuner._create_evaluator_function(evaluator_path, tmpdir)
            assert os.path.exists(evaluator_path)

            # Create config yaml
            config_path = os.path.join(tmpdir, "config.yaml")
            tuner._create_config_yaml(config_path)
            assert os.path.exists(config_path)

            # Verify we can import and use the evaluator
            evaluator_content = Path(evaluator_path).read_text()
            assert "def evaluate(program_path):" in evaluator_content
            assert "def validate_config(config):" in evaluator_content
