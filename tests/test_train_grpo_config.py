import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_grpo import instantiate_with_supported_kwargs


class SmallConfig:
    def __init__(self, output_dir: str, learning_rate: float = 1e-6):
        self.output_dir = output_dir
        self.learning_rate = learning_rate


class TrainGrpoConfigTests(unittest.TestCase):
    def test_filters_unsupported_kwargs(self):
        config = instantiate_with_supported_kwargs(
            SmallConfig,
            {
                "output_dir": "outputs/test",
                "learning_rate": 5e-6,
                "max_prompt_length": 2048,
            },
            "SmallConfig",
        )
        self.assertEqual(config.output_dir, "outputs/test")
        self.assertEqual(config.learning_rate, 5e-6)


if __name__ == "__main__":
    unittest.main()
