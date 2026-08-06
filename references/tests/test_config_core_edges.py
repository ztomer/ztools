"""Additional tests for lib.config_core covering specific uncovered branches.

Coverage gaps targeted:
- Lines 63-64: ConfigurationError raised when load_config fails in _auto_load
- Lines 72-77: User config overlay when USER_CONFIG_PATH exists
- Line 92: loaded = {} when init_config gets None
- Line 94: ValueError when config is not a dict
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_config():
    import lib.config_core as cc

    cc._config_loaded = False
    cc._config = {}
    cc._model_configs_cache = {}
    yield
    cc._config_loaded = False
    cc._config = {}
    cc._model_configs_cache = {}


class TestAutoLoadConfigurationError:
    """Lines 63-64: _auto_load raises ConfigurationError when load_config fails."""

    def test_auto_load_raises_on_corrupt_config(self, tmp_path):
        import lib.config_core as cc

        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        config_toml = conf_dir / "config.toml"
        config_toml.write_text("valid = true\n")

        with (
            patch("lib.config_core.Path") as mock_path,
            patch("lib.config_core.load_config", side_effect=RuntimeError("parse error")),
        ):
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value = conf_dir
            mock_path.return_value = instance
            with pytest.raises(cc.ConfigurationError, match="parse error"):
                cc._auto_load()


class TestAutoLoadUserConfigOverlay:
    """Lines 72-77: user config overlay when USER_CONFIG_PATH exists."""

    def test_user_config_overlay_applied(self, tmp_path):
        import lib.config_core as cc

        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        config_toml = conf_dir / "config.toml"
        config_toml.write_text('[timeouts]\njson = 100\n')

        user_config = tmp_path / "user_config.toml"
        user_config.write_text('[timeouts]\njson = 999\n')

        with (
            patch("lib.config_core.Path") as mock_path,
            patch.object(cc, "USER_CONFIG_PATH", user_config),
        ):
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value = conf_dir
            mock_path.return_value = instance
            cc._auto_load()

        assert cc._config.get("timeouts", {}).get("json") == 999

    def test_user_config_overlay_error_is_swallowed(self, tmp_path, capsys):
        import lib.config_core as cc

        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        config_toml = conf_dir / "config.toml"
        config_toml.write_text('[timeouts]\njson = 100\n')

        user_config = tmp_path / "bad_user_config.toml"
        user_config.write_text("not valid toml [[[")

        with (
            patch("lib.config_core.Path") as mock_path,
            patch.object(cc, "USER_CONFIG_PATH", user_config),
        ):
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value = conf_dir
            mock_path.return_value = instance
            cc._auto_load()

        # Base config still loaded despite user config error
        assert cc._config_loaded is True
        out = capsys.readouterr().out
        assert "Failed to read user config" in out

    def test_user_config_overlay_non_dict_ignored(self, tmp_path):
        import lib.config_core as cc

        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        config_toml = conf_dir / "config.toml"
        config_toml.write_text('[timeouts]\njson = 100\n')

        user_config = tmp_path / "user_config.toml"
        user_config.write_text('[timeouts]\njson = 100\n')

        def fake_load(path):
            if "user" in str(path):
                return "not a dict"
            import tomllib
            with open(path, "rb") as f:
                return tomllib.load(f)

        with (
            patch("lib.config_core.Path") as mock_path,
            patch.object(cc, "USER_CONFIG_PATH", user_config),
            patch("lib.config_core.load_config", side_effect=fake_load),
        ):
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value = conf_dir
            mock_path.return_value = instance
            cc._auto_load()

        # Base config loaded, user config non-dict is silently ignored
        assert cc._config_loaded is True


class TestInitConfigEdgeCases:
    """Lines 92, 94: init_config edge cases."""

    def test_init_config_none_result_becomes_empty_dict(self, tmp_path):
        import lib.config_core as cc

        toml_file = tmp_path / "empty.toml"
        toml_file.write_text("")

        with patch("lib.config_core.load_config", return_value=None):
            cc.init_config(str(toml_file))

        assert cc._config == {}
        assert cc._config_loaded is True

    def test_init_config_non_dict_raises_value_error(self, tmp_path):
        import lib.config_core as cc

        toml_file = tmp_path / "list.toml"
        toml_file.write_text("x = 1\n")

        with patch("lib.config_core.load_config", return_value=[1, 2, 3]):
            with pytest.raises(ValueError, match="Config must be a dictionary"):
                cc.init_config(str(toml_file))
