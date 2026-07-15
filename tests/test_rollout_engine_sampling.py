import unittest

import torch

from trainer.rollout_engine import TorchRolloutEngine


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generation_kwargs = None

    def generate(self, input_ids, **kwargs):
        self.generation_kwargs = kwargs
        next_token = input_ids.new_full((input_ids.shape[0], 1), 2)
        return torch.cat([input_ids, next_token], dim=1)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    @staticmethod
    def batch_decode(completion_ids, skip_special_tokens=True):
        return ["" for _ in completion_ids]


class RolloutSamplingCompatibilityTest(unittest.TestCase):
    def test_omitted_sampling_controls_preserve_old_generate_defaults(self):
        model = RecordingModel()
        engine = TorchRolloutEngine(model, FakeTokenizer(), device="cpu")
        result = engine.rollout(
            torch.tensor([[1, 3]]),
            torch.tensor([[1, 1]]),
            num_generations=1,
            max_new_tokens=1,
            calculate_logps=False,
        )
        self.assertNotIn("top_p", model.generation_kwargs)
        self.assertNotIn("top_k", model.generation_kwargs)
        self.assertEqual(tuple(result.per_token_logps.shape), (1, 1))
        self.assertEqual(result.per_token_logps.item(), 0.0)

    def test_explicit_sampling_controls_reach_generate(self):
        model = RecordingModel()
        engine = TorchRolloutEngine(model, FakeTokenizer(), device="cpu")
        engine.rollout(
            torch.tensor([[1, 3]]),
            torch.tensor([[1, 1]]),
            num_generations=1,
            max_new_tokens=1,
            top_p=1.0,
            top_k=0,
            calculate_logps=False,
        )
        self.assertEqual(model.generation_kwargs["top_p"], 1.0)
        self.assertEqual(model.generation_kwargs["top_k"], 0)


if __name__ == "__main__":
    unittest.main()
