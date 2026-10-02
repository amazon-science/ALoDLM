"""Check uniform gate initialization and weight restoration. Run: python -m unittest discover -s tests -p test_gate.py."""

import tempfile
import unittest

import torch

from alodlm.config import ModelConfig
from alodlm.model import ALoDLM, GateHead
from helpers import tiny_model


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_uniform_exit_distribution_at_both_model_widths(self):
        for hidden_size in (2048, 4096):
            with self.subTest(hidden_size=hidden_size):
                gate = GateHead(hidden_size, ModelConfig().gate_init_pi)
                features = torch.randn(3, hidden_size)
                survival = torch.ones(3)
                masses = []
                for depth in range(3):
                    hazard = gate(features, depth).sigmoid()
                    masses.append(survival * hazard)
                    survival = survival * (1 - hazard)
                masses.append(survival)
                torch.testing.assert_close(torch.stack(masses), torch.full((4, 3), 0.25))

    def test_loading_restores_learned_gate_over_uniform_initialization(self):
        model = tiny_model()
        with torch.no_grad():
            model.exit_gate.net.weight.fill_(0.125)
            model.exit_gate.net.bias.fill_(-0.25)
            model.exit_gate.depth_bias.copy_(torch.tensor([-0.7, 0.2, 0.8, 0.0]))
        learned = {key: value.clone() for key, value in model.exit_gate.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = ALoDLM.from_pretrained(directory)
        self.assertEqual(tuple(restored.config.gate_init_pi), (0.25,) * 4)
        for key, value in restored.exit_gate.state_dict().items():
            torch.testing.assert_close(value, learned[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
