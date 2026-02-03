"""Tests for curriculum learning callback."""

from unittest.mock import MagicMock, PropertyMock, patch

from src.callbacks.curriculum import DEFAULT_CURRICULUM, CurriculumCallback, CurriculumLevel


class TestCurriculumLevels:
    def test_default_curriculum_has_levels(self):
        assert len(DEFAULT_CURRICULUM) >= 3

    def test_levels_have_increasing_speed(self):
        for i in range(1, len(DEFAULT_CURRICULUM)):
            assert DEFAULT_CURRICULUM[i].speed_max >= DEFAULT_CURRICULUM[i - 1].speed_max

    def test_terminal_level_has_zero_threshold(self):
        assert DEFAULT_CURRICULUM[-1].threshold == 0.0

    def test_first_level_is_stationary(self):
        assert DEFAULT_CURRICULUM[0].speed_min == 0.0
        assert DEFAULT_CURRICULUM[0].speed_max == 0.0


class TestCurriculumCallback:
    def _make_callback(self, eval_window=1, levels=None):
        """Create a callback with mocked parent and model."""
        cb = CurriculumCallback(
            levels=levels, eval_window=eval_window, shots_per_episode=50, verbose=0
        )
        cb.parent = MagicMock()
        cb.model = MagicMock()
        # SB3 BaseCallback.logger is a property — mock it via the instance's __dict__
        mock_logger = MagicMock()
        cb._logger = mock_logger
        return cb

    def test_starts_at_level_zero(self):
        cb = self._make_callback()
        assert cb.current_level_idx == 0

    def test_does_not_advance_below_threshold(self):
        cb = self._make_callback(eval_window=1)
        cb.parent.last_mean_reward = 5.0  # low reward → low hit rate
        with patch.object(type(cb), "logger", new_callable=PropertyMock) as mock_log:
            mock_log.return_value = MagicMock()
            cb._on_step()
        assert cb.current_level_idx == 0

    def test_advances_when_threshold_met(self):
        levels = [
            CurriculumLevel("level0", 0.0, 0.0, 0.50),
            CurriculumLevel("level1", 1.0, 2.0, 0.0),
        ]
        cb = self._make_callback(eval_window=1, levels=levels)
        cb.parent.last_mean_reward = 50.0
        cb.parent.eval_env = MagicMock()
        with patch.object(type(cb), "logger", new_callable=PropertyMock) as mock_log:
            mock_log.return_value = MagicMock()
            cb._on_step()
        assert cb.current_level_idx == 1

    def test_requires_eval_window_consecutive(self):
        levels = [
            CurriculumLevel("level0", 0.0, 0.0, 0.50),
            CurriculumLevel("level1", 1.0, 2.0, 0.0),
        ]
        cb = self._make_callback(eval_window=3, levels=levels)
        cb.parent.eval_env = MagicMock()

        with patch.object(type(cb), "logger", new_callable=PropertyMock) as mock_log:
            mock_log.return_value = MagicMock()

            # First two evals above threshold
            cb.parent.last_mean_reward = 50.0
            cb._on_step()
            assert cb.current_level_idx == 0
            cb._on_step()
            assert cb.current_level_idx == 0

            # Drop below threshold — resets counter
            cb.parent.last_mean_reward = 5.0
            cb._on_step()
            assert cb.current_level_idx == 0

            # Need 3 consecutive again
            cb.parent.last_mean_reward = 50.0
            cb._on_step()
            cb._on_step()
            assert cb.current_level_idx == 0
            cb._on_step()
            assert cb.current_level_idx == 1

    def test_does_not_advance_past_terminal(self):
        levels = [
            CurriculumLevel("level0", 0.0, 0.0, 0.50),
            CurriculumLevel("level1", 1.0, 2.0, 0.0),  # terminal
        ]
        cb = self._make_callback(eval_window=1, levels=levels)
        cb.parent.eval_env = MagicMock()
        cb.parent.last_mean_reward = 50.0
        with patch.object(type(cb), "logger", new_callable=PropertyMock) as mock_log:
            mock_log.return_value = MagicMock()
            cb._on_step()  # advance to level 1
            assert cb.current_level_idx == 1
            cb._on_step()  # should stay at level 1
            assert cb.current_level_idx == 1

    def test_env_method_called_on_advance(self):
        levels = [
            CurriculumLevel("level0", 0.0, 0.0, 0.50),
            CurriculumLevel("level1", 1.0, 2.0, 0.0),
        ]
        cb = self._make_callback(eval_window=1, levels=levels)
        cb.parent.eval_env = MagicMock()
        cb.parent.last_mean_reward = 50.0
        with patch.object(type(cb), "logger", new_callable=PropertyMock) as mock_log:
            mock_log.return_value = MagicMock()
            cb._on_step()

        # Verify env_method was called on both training and eval envs
        cb.model.get_env().env_method.assert_called_with(
            "set_curriculum_level", 1.0, 2.0
        )
        cb.parent.eval_env.env_method.assert_called_with(
            "set_curriculum_level", 1.0, 2.0
        )

    def test_logs_to_tensorboard(self):
        cb = self._make_callback()
        cb.parent.last_mean_reward = 25.0
        with patch.object(type(cb), "logger", new_callable=PropertyMock) as mock_log:
            logger_mock = MagicMock()
            mock_log.return_value = logger_mock
            cb._on_step()
        logger_mock.record.assert_any_call("curriculum/level", 0)
        logger_mock.record.assert_any_call("curriculum/level_name", "stationary")
