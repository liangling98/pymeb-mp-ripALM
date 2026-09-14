import unittest

import torch

from experiments.real_datasets import group_ball_tensors, standardize_features


class RealDatasetConstructionTests(unittest.TestCase):
    def test_standardization_handles_constant_columns(self):
        features = torch.tensor(
            [[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]], dtype=torch.float64
        )
        standardized, mean, scale = standardize_features(features)

        self.assertTrue(
            torch.allclose(mean, torch.tensor([3.0, 2.0], dtype=torch.float64))
        )
        self.assertEqual(float(scale[1]), 1.0)
        self.assertTrue(
            torch.allclose(
                standardized.mean(dim=0), torch.zeros(2, dtype=torch.float64)
            )
        )
        self.assertTrue(
            torch.equal(standardized[:, 1], torch.zeros(3, dtype=torch.float64))
        )

    def test_group_balls_enclose_every_source_point(self):
        pixels = torch.tensor(
            [[0, 0], [2, 0], [0, 3], [0, 5]], dtype=torch.uint8
        )
        group_ids = torch.tensor([0, 0, 1, 1])
        centers, radii, sizes, labels = group_ball_tensors(
            pixels,
            group_ids,
            group_labels=["a", "b"],
        )

        scaled = pixels.to(torch.float64) / 255.0
        for group_id in range(2):
            distances = torch.linalg.vector_norm(
                scaled[group_ids == group_id] - centers[group_id], dim=1
            )
            self.assertLessEqual(float(distances.max()), float(radii[group_id]) + 1e-15)
        self.assertEqual(sizes.tolist(), [2, 2])
        self.assertEqual(labels, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
