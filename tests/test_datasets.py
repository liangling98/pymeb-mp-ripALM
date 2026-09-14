import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import torch

from experiments import datasets
from experiments.datasets import paper_meb_problem


class PaperMEBDatasetTests(unittest.TestCase):
    def test_first_values_match_paper_recurrence_order(self):
        problem = paper_meb_problem(2, 3)

        expected_radii = torch.tensor([76.07421875, 27.1484375], dtype=torch.float64)
        expected_centers = torch.tensor(
            [
                [53.0517578125, 8.056640625, 85.2294921875],
                [81.0791015625, 80.224609375, 99.9755859375],
            ],
            dtype=torch.float64,
        )

        self.assertTrue(torch.allclose(problem.radii, expected_radii))
        self.assertTrue(torch.allclose(problem.centers, expected_centers))

    def test_dtype_and_shape(self):
        problem = paper_meb_problem(5, 2, dtype=torch.float32)

        self.assertEqual(problem.m, 5)
        self.assertEqual(problem.d, 2)
        self.assertEqual(problem.dtype, torch.float32)
        self.assertTrue(torch.equal(problem.original_indices, torch.arange(5)))

    def test_generated_problem_is_cached(self):
        with TemporaryDirectory() as tmpdir:
            old_cache_dir = datasets.DATASET_CACHE_DIR
            datasets.DATASET_CACHE_DIR = Path(tmpdir)
            try:
                first = paper_meb_problem(4, 2, dtype=torch.float32)
                cache_files = list(Path(tmpdir).glob("paper_meb_m4_d2_float32.pt"))
                self.assertEqual(len(cache_files), 1)

                cached_values = torch.load(
                    cache_files[0], map_location="cpu", weights_only=True
                )
                cached_values[0, 0] = 123.0
                torch.save(cached_values, cache_files[0])

                second = paper_meb_problem(4, 2, dtype=torch.float32)
            finally:
                datasets.DATASET_CACHE_DIR = old_cache_dir

        self.assertTrue(torch.allclose(first.centers, second.centers))
        self.assertEqual(float(second.radii[0]), 123.0)


if __name__ == "__main__":
    unittest.main()
