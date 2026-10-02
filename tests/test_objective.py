"""Check loss, gradients, and attention boundaries. Run: python -m unittest discover -s tests."""

from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from alodlm.batch import build_batch
from alodlm.attention import sparse_plan
from alodlm.config import ModelConfig
from alodlm.credit import masked_sequence_ids, sequence_score_function
from alodlm.loss import autoregressive_loss, outcome_loss
from helpers import tiny_model


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sequence_credit_gradient_and_isolation(self):
        credit = torch.tensor([1., 2., 4.], requires_grad=True)
        logp = torch.tensor([-.2, -.3, -.7], requires_grad=True)
        result = sequence_score_function(credit, logp, torch.tensor([0, 0, 1]), 2)
        grad = torch.autograd.grad(result, (credit, logp), allow_unused=True)
        self.assertIsNone(grad[0])
        torch.testing.assert_close(grad[1], torch.tensor([1., 1., 4 / 3]))
        with self.assertRaises(ValueError):
            masked_sequence_ids(torch.tensor([True, False]), torch.tensor([0, 1]))

    def test_full_outcome_loss_matches_independent_analytic_gradient(self):
        cfg = ModelConfig(max_depth=3, gate_init_pi=(1 / 3,) * 3)
        ce = torch.tensor([[3., 4., 5., 6.], [2., 3., 3., 4.], [1., 2., 2., 3.]],
                          requires_grad=True)
        gates = [torch.tensor([-.4, .2, -.3, .7], requires_grad=True) for _ in range(3)]
        depths = torch.tensor([1, 2, 3, 2])
        batch = SimpleNamespace(masked=torch.ones(4, dtype=torch.bool),
                                mask_probability=torch.tensor([.2, .4, .5, .8]),
                                boundaries=torch.tensor([0, 2, 4]))
        actual, _ = outcome_loss(ce, gates, depths, batch, cfg)
        probs = torch.stack(gates).sigmoid()
        logq, full = [], []
        lm = ce.new_zeros(())
        weights = 1 / (batch.mask_probability + 1e-8)
        weights /= weights.sum()
        for i in range(4):
            z = int(depths[i])
            survival = ce.new_tensor(1.)
            masses = []
            for d in range(3):
                masses.append(survival if d == 2 else survival * probs[d, i])
                survival = survival * (1 - probs[d, i])
            full.append(torch.stack(masses))
            logq.append(sum(F.logsigmoid(-gates[d][i]) for d in range(z - 1))
                        + (F.logsigmoid(gates[z - 1][i]) if z < 3 else 0))
            survival = ce.new_tensor(1.)
            for d in range(z):
                mass = survival if d == z - 1 else survival * probs[d, i]
                lm = lm + weights[i] * mass.detach() * ce[d, i]
                survival = survival * (1 - probs[d, i])
        logq = torch.stack(logq)
        qbar = torch.stack(full, 1).mean(1).detach()
        prior = torch.softmax(-cfg.depth_prior_c * torch.arange(1, 4).float(), 0)
        sampled_ce = ce[depths - 1, torch.arange(4)]
        cost = ((sampled_ce - ce[0]).detach()
                + cfg.kl_beta_mi * (logq.detach() - qbar[depths - 1].log())
                + cfg.kl_beta_marg * (qbar[depths - 1].log() - prior[depths - 1].log()))
        expected = lm + (cost[:2].sum() * logq[:2].sum() + cost[2:].sum() * logq[2:].sum()) / 4
        torch.testing.assert_close(actual, expected)
        ag = torch.autograd.grad(actual, gates + [ce], retain_graph=True)
        eg = torch.autograd.grad(expected, gates + [ce], allow_unused=True)
        for a, e in zip(ag, eg):
            torch.testing.assert_close(a, e if e is not None else torch.zeros_like(a))

    def test_ar_segment_reduction_and_gradient(self):
        torch.manual_seed(3)
        logits = torch.randn(16, 16, requires_grad=True)
        labels = torch.arange(8)
        labels[1] = -100
        actual = autoregressive_loss(logits, labels, torch.tensor([0, 8, 16]))
        expected = F.cross_entropy(logits[[1, 2, 8, 9, 10]], labels[[2, 3, 5, 6, 7]])
        torch.testing.assert_close(actual, expected)
        ag = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        eg = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(ag, eg)

    def test_packed_attention_does_not_cross_sequences(self):
        ids = torch.arange(8)
        batch = build_batch(ids, ids.clone(), torch.tensor([0, 4, 8]), 2, 3)
        self.assertFalse(batch.attention_mask[:8, 8:].any())
        self.assertFalse(batch.attention_mask[8:, :8].any())
        self.assertFalse(batch.attention_mask[4, 2])
        self.assertTrue(batch.attention_mask[6, 0])
        self.assertFalse(batch.attention_mask[0, 4])

    def test_sparse_ranges_match_dense_attention(self):
        ids = torch.arange(11)
        batch = build_batch(ids, ids.clone(), torch.tensor([0, 4, 11]), 3, 3)
        plan = sparse_plan(batch.boundaries.tolist(), 3, "magi")
        materialized = torch.zeros_like(batch.attention_mask)
        for (qa, qb), (ka, kb), causal in zip(plan["q_ranges"], plan["k_ranges"], plan["attn_type_map"]):
            allowed = torch.ones(qb - qa, kb - ka, dtype=torch.bool)
            materialized[qa:qb, ka:kb] |= allowed.tril() if causal else allowed
        self.assertTrue(torch.equal(materialized, batch.attention_mask))

    def test_real_model_updates_both_gate_and_denoiser(self):
        model = tiny_model()
        ids = torch.tensor([4, 7, 9, 2, 5, 9, 12, 2])
        labels = ids.clone()
        labels[:4] = -100
        gate_before = model.exit_gate.net.weight.detach().clone()
        word_before = model.backbone.model.embed_tokens.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        torch.manual_seed(5)
        loss, metrics = model(ids, labels, torch.tensor([0, 8]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        self.assertFalse(torch.equal(model.exit_gate.net.weight, gate_before))
        self.assertFalse(torch.equal(model.backbone.model.embed_tokens.weight, word_before))
        self.assertIn("sampled_mean_depth", metrics)


if __name__ == "__main__":
    unittest.main()
