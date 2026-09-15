from __future__ import annotations

import math
import unittest

import numpy as np

from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.envs import PaperPuffPlume


FAST_PAPER = {
    "plume_model": "paper",
    "scenario_mode": "fixed",
    "paper_plume_profile": "constant",
    "plume_overrides": {"puff_warmup_steps": 1},
}


class PaperPuffPlumeTests(unittest.TestCase):
    def test_reference_radius_concentration_law_and_square_support(self) -> None:
        plume = PaperPuffPlume(plume_overrides={"puff_warmup_steps": 0})
        plume.puffs = np.array([[0.0, 0.0, 0.02]], dtype=np.float64)

        self.assertAlmostEqual(plume.concentration(0.0, 0.0, add_noise=False), 0.125)
        self.assertAlmostEqual(plume.concentration(0.019, 0.019, add_noise=False), 0.125)
        self.assertEqual(plume.concentration(0.021, 0.0, add_noise=False), 0.0)

    def test_reference_profiles_change_birth_and_diffusion_rates(self) -> None:
        constant = PaperPuffPlume(profile="constant")
        sparse = PaperPuffPlume(profile="sparse")
        sparser = PaperPuffPlume(profile="sparser")

        self.assertEqual(constant.puff_birth_rate, 1.0)
        self.assertEqual(sparse.puff_birth_rate, 0.4)
        self.assertEqual(sparser.puff_birth_rate, 0.4)
        self.assertEqual(constant.puff_spread_rate, 0.01)
        self.assertEqual(sparser.puff_spread_rate, 0.005)

    def test_seed_reproduces_online_plume(self) -> None:
        kwargs = {"plume_overrides": {"puff_warmup_steps": 1}}
        first = PaperPuffPlume(**kwargs)
        second = PaperPuffPlume(**kwargs)
        first.reset(seed=123)
        second.reset(seed=123)
        for _ in range(4):
            first.advance()
            second.advance()

        np.testing.assert_array_equal(first.puffs, second.puffs)
        self.assertEqual(first.wind_direction, second.wind_direction)

    def test_grid_matches_point_concentration(self) -> None:
        plume = PaperPuffPlume(plume_overrides={"puff_warmup_steps": 0})
        plume.puffs = np.array(
            [[0.0, 0.0, 0.08], [0.03, 0.0, 0.04]], dtype=np.float64
        )
        xs, ys, field = plume.grid(resolution=101, add_noise=False)
        ix = int(np.argmin(np.abs(xs)))
        iy = int(np.argmin(np.abs(ys)))

        self.assertAlmostEqual(
            float(field[iy, ix]),
            plume.concentration(float(xs[ix]), float(ys[iy]), add_noise=False),
            places=6,
        )

    def test_switch_once_turns_wind_by_45_degrees(self) -> None:
        plume = PaperPuffPlume(
            profile="switch_once",
            dt=0.02,
            plume_overrides={
                "puff_warmup_steps": 0,
                "switch_once_time_s": 0.01,
            },
        )
        plume.reset(seed=1)
        plume.advance()
        self.assertAlmostEqual(plume.wind_direction, math.pi / 4.0)

    def test_mobile_environment_selects_paper_backend_without_space_changes(self) -> None:
        paper = MobileWhiskerPuffEnv(FAST_PAPER)
        dynamic = MobileWhiskerPuffEnv(
            {"scenario_mode": "fixed", "plume_overrides": {"puff_warmup_steps": 1}}
        )
        try:
            obs, info = paper.reset(seed=7)
            self.assertIsInstance(paper.plume, PaperPuffPlume)
            self.assertEqual(paper.plume.metadata()["paper_plume_profile"], "constant")
            self.assertEqual(obs.shape, dynamic.observation_space.shape)
            self.assertEqual(paper.action_space, dynamic.action_space)
            self.assertEqual(info["source"], (-0.8, 0.0))
        finally:
            paper.close()
            dynamic.close()

    def test_invalid_backend_and_profile_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "plume_model"):
            MobileWhiskerPuffEnv({"plume_model": "unknown"})
        with self.assertRaisesRegex(ValueError, "paper plume profile"):
            PaperPuffPlume(profile="unknown")


if __name__ == "__main__":
    unittest.main()
