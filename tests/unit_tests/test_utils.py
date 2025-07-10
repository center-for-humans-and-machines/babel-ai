"""Tests for the utils module."""

import pytest
from pydantic import BaseModel

from utils import load_yaml_config


class ConfigModel(BaseModel):
    """Test configuration model with various field types."""

    name: str
    value: int
    enabled: bool = True
    settings: dict = {}
    items: list = []


class TestLoadYamlConfig:
    """Test the load_yaml_config function."""

    def test_load_yaml_config_success(self, tmp_path):
        """Test successful loading of YAML configuration."""
        # Create test YAML file with all field types
        config_file = tmp_path / "test.yaml"
        config_content = """
name: "test_config"
value: 42
enabled: false
settings:
  debug: true
  timeout: 30
items:
  - "item1"
  - "item2"
"""
        config_file.write_text(config_content.strip())

        # Load configuration
        config = load_yaml_config(ConfigModel, str(config_file))

        # Verify all fields
        assert isinstance(config, ConfigModel)
        assert config.name == "test_config"
        assert config.value == 42
        assert config.enabled is False
        assert config.settings == {"debug": True, "timeout": 30}
        assert config.items == ["item1", "item2"]

    def test_load_yaml_config_with_defaults(self, tmp_path):
        """Test loading config that uses default values."""
        config_file = tmp_path / "minimal.yaml"
        config_content = """
name: "minimal_config"
value: 100
"""
        config_file.write_text(config_content.strip())

        config = load_yaml_config(ConfigModel, str(config_file))

        assert config.name == "minimal_config"
        assert config.value == 100
        assert config.enabled is True  # Default value
        assert config.settings == {}  # Default value
        assert config.items == []  # Default value

    def test_load_yaml_config_pathlib_path(self, tmp_path):
        """Test loading config using pathlib.Path object."""
        config_file = tmp_path / "path_test.yaml"
        config_content = """
name: "path_test"
value: 200
"""
        config_file.write_text(config_content.strip())

        # Use Path object instead of string
        config = load_yaml_config(ConfigModel, config_file)

        assert config.name == "path_test"
        assert config.value == 200

    def test_load_yaml_config_file_not_found(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config(ConfigModel, "nonexistent.yaml")

    def test_load_yaml_config_validation_error(self, tmp_path):
        """Test validation error when data doesn't match schema."""
        config_file = tmp_path / "invalid.yaml"
        config_content = """
name: "test"
value: "not_a_number"  # Should be int
"""
        config_file.write_text(config_content.strip())

        # Pydantic will raise ValidationError
        with pytest.raises(Exception):
            load_yaml_config(ConfigModel, str(config_file))
