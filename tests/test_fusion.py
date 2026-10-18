import unittest
from unittest.mock import patch

import numpy as np

from traversalmap.core import make_landmark, run_case, update_landmark, wrap


def observation(s, angle):
    return dict(s=s, angle=angle, roll=0, variance_s=1, variance_angle=0.04,
                descriptor=[1., 0.])


class FusionTests(unittest.TestCase):
    def test_independent_updates_coordinates_and_uncertainty(self):
        landmark = make_landmark(observation(0, 3.13))
        update_landmark(landmark, observation(2, -3.13), independent=True)
        self.assertAlmostEqual(landmark["s"], 1)
        self.assertAlmostEqual(landmark["variance_s"], 0.5)
        self.assertAlmostEqual(landmark["variance_angle"], 0.02)
        self.assertAlmostEqual(abs(landmark["angle"]), np.pi)
        self.assertEqual(landmark["observations"], 2)

    def test_unknown_correlation_does_not_halve_variance(self):
        landmark = make_landmark(observation(0, 0))
        update_landmark(landmark, observation(2, 0))
        self.assertAlmostEqual(landmark["variance_s"], 1)
        self.assertAlmostEqual(landmark["variance_angle"], 0.04)

    def test_roll_noise_is_part_of_corrected_angle_variance(self):
        value = observation(0, 3.13)
        value.update(roll=0.05, variance_roll=0.0025, truth_id="never-used")
        landmark = make_landmark(value)
        self.assertAlmostEqual(landmark["angle"], wrap(3.18))
        self.assertAlmostEqual(landmark["variance_angle"], 0.0425)
        self.assertNotIn("truth_id", landmark)

    def test_association_enters_fusion_and_fixed_map_is_a_control(self):
        case = dict(independent_observations=True, traversals=[
            dict(direction=1, observations=[observation(0, 0)]),
            dict(direction=-1, observations=[observation(2, 0)])])
        with patch("traversalmap.core.optimize", return_value=(np.array([0]), np.array([[1., 0.]]))):
            fused = run_case(case, "traversalmap")
            fixed = run_case(case, "fixed_map")
        self.assertEqual(fused["landmarks"][0]["s"], 1)
        self.assertEqual(fused["landmarks"][0]["observations"], 2)
        self.assertEqual(fixed["landmarks"][0]["s"], 0)


if __name__ == "__main__":
    unittest.main()
