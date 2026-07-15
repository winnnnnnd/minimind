import unittest

import torch

from trainer.opd_utils import (
    build_completion_mask,
    compute_opd_loss,
    compute_reference_kl_loss,
    extract_incremental_observation_suffix,
    is_reference_replay_step,
)


class OPDUtilsTest(unittest.TestCase):
    def test_incremental_suffix_preserves_noncanonical_sampled_action_tokens(self):
        # The sampled action uses ["Ġ3", "6"], while the canonical template
        # uses the merged ["Ġ36"]. Only the post-EOS suffix is appended.
        sampled_action = [707, 57, 2]
        closed_assistant = [10, 6285, 2, 234]
        observed_context = closed_assistant + [20, 21, 30]
        suffix = extract_incremental_observation_suffix(
            closed_assistant, observed_context, sampled_action, eos_token_id=2
        )
        self.assertEqual(suffix, [234, 20, 21, 30])
        self.assertEqual(sampled_action + suffix, [707, 57, 2, 234, 20, 21, 30])

    def test_incremental_suffix_supplies_masked_eos_when_action_was_truncated(self):
        sampled_action = [707, 57]
        closed_assistant = [10, 6285, 2, 234]
        observed_context = closed_assistant + [20, 21]
        suffix = extract_incremental_observation_suffix(
            closed_assistant, observed_context, sampled_action, eos_token_id=2
        )
        self.assertEqual(suffix, [2, 234, 20, 21])

    def test_incremental_suffix_rejects_a_changed_canonical_prefix(self):
        with self.assertRaisesRegex(ValueError, "not a prefix"):
            extract_incremental_observation_suffix(
                [10, 6285, 2, 234],
                [10, 707, 57, 2, 234, 20],
                [707, 57, 2],
                eos_token_id=2,
            )

    def test_reference_replay_schedule_is_even_and_deterministic(self):
        replayed = [step for step in range(1, 13) if is_reference_replay_step(step, 0.25)]
        self.assertEqual(replayed, [4, 8, 12])
        self.assertTrue(all(is_reference_replay_step(step, 1.0) for step in range(1, 5)))

    def test_identical_reference_policies_have_zero_loss_and_gradient(self):
        student = torch.tensor([[[0.2, -0.1, 0.7]]], requires_grad=True)
        reference = student.detach().clone()
        loss, metrics = compute_reference_kl_loss(
            student, reference, torch.tensor([[1]])
        )
        loss.backward()
        self.assertAlmostEqual(loss.item(), 0.0, places=7)
        self.assertTrue(torch.allclose(student.grad, torch.zeros_like(student.grad), atol=1e-7))
        self.assertAlmostEqual(metrics["reference_top1_agreement"].item(), 1.0, places=7)

    def test_reference_kl_gradient_moves_student_toward_reference(self):
        student = torch.zeros(1, 1, 3, requires_grad=True)
        reference = torch.tensor([[[-2.0, 2.0, -2.0]]])
        loss, metrics = compute_reference_kl_loss(
            student, reference, torch.tensor([[1]])
        )
        loss.backward()
        self.assertLess(student.grad[0, 0, 1].item(), 0.0)
        self.assertGreater(student.grad[0, 0, 0].item(), 0.0)
        self.assertGreater(metrics["reference_reverse_kl"].item(), 0.0)

    def test_reference_kl_masks_non_assistant_tokens(self):
        student = torch.zeros(1, 2, 3, requires_grad=True)
        reference = torch.tensor([[[-2.0, 2.0, -2.0], [2.0, -2.0, -2.0]]])
        loss, _ = compute_reference_kl_loss(
            student, reference, torch.tensor([[1, 0]])
        )
        loss.backward()
        self.assertTrue(torch.allclose(student.grad[0, 1], torch.zeros(3), atol=1e-7))

    def test_empty_reference_mask_is_finite_and_has_zero_gradient(self):
        student = torch.zeros(1, 1, 3, requires_grad=True)
        reference = torch.tensor([[[-2.0, 2.0, -2.0]]])
        loss, metrics = compute_reference_kl_loss(
            student, reference, torch.tensor([[0]])
        )
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.allclose(student.grad, torch.zeros_like(student.grad)))
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))

    def test_completion_mask_stops_after_first_eos_and_removes_padding(self):
        completion_ids = torch.tensor([[4, 2, 2, 2], [5, 6, 0, 0], [7, 2, 9, 0]])
        base_mask = torch.ones_like(completion_ids)
        actual = build_completion_mask(completion_ids, base_mask, eos_token_id=2, pad_token_id=0)
        expected = torch.tensor(
            [[True, True, False, False], [True, True, False, False], [True, True, False, False]]
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_completion_mask_respects_variable_length_base_mask(self):
        completion_ids = torch.tensor([[3, 4, 0], [5, 2, 0]])
        base_mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
        actual = build_completion_mask(completion_ids, base_mask, eos_token_id=2, pad_token_id=0)
        self.assertTrue(torch.equal(actual, base_mask.bool()))

    def test_identical_topk_policies_have_zero_loss_and_gradient(self):
        student = torch.tensor([[[0.2, -0.1, 0.7]]], requires_grad=True)
        teacher = student.detach().clone()
        loss, metrics = compute_opd_loss(
            student, teacher, torch.tensor([[2]]), torch.tensor([[1]]), distill_top_k=2
        )
        loss.backward()
        self.assertAlmostEqual(loss.item(), 0.0, places=7)
        self.assertTrue(torch.allclose(student.grad, torch.zeros_like(student.grad), atol=1e-7))
        self.assertAlmostEqual(metrics["reverse_kl"].item(), 0.0, places=7)
        self.assertAlmostEqual(metrics["topk_overlap_ratio"].item(), 1.0, places=7)

    def test_topk_gradient_moves_student_toward_teacher(self):
        student = torch.zeros(1, 1, 3, requires_grad=True)
        teacher = torch.tensor([[[-2.0, 2.0, -2.0]]])
        loss, metrics = compute_opd_loss(
            student, teacher, torch.tensor([[1]]), torch.tensor([[1]]), distill_top_k=3
        )
        loss.backward()
        self.assertLess(student.grad[0, 0, 1].item(), 0.0)
        self.assertGreater(student.grad[0, 0, 0].item(), 0.0)
        self.assertGreater(metrics["reverse_kl"].item(), 0.0)

    def test_sampled_k1_gradient_favors_teacher_preferred_sample(self):
        student = torch.zeros(1, 1, 3, requires_grad=True)
        teacher = torch.tensor([[[-2.0, 2.0, -2.0]]])
        loss, _ = compute_opd_loss(
            student, teacher, torch.tensor([[1]]), torch.tensor([[1]]), distill_top_k=0
        )
        loss.backward()
        self.assertLess(student.grad[0, 0, 1].item(), 0.0)

    def test_full_vocabulary_mode_reports_nonnegative_reverse_kl(self):
        student = torch.tensor([[[1.0, 0.0, -1.0]]], requires_grad=True)
        teacher = torch.tensor([[[0.0, 1.0, -1.0]]])
        loss, metrics = compute_opd_loss(
            student, teacher, torch.tensor([[0]]), torch.tensor([[1]]), distill_top_k=-1
        )
        loss.backward()
        self.assertGreaterEqual(metrics["reverse_kl"].item(), -1e-7)
        self.assertAlmostEqual(metrics["topk_overlap_ratio"].item(), 1.0, places=7)

    def test_full_vocabulary_surrogate_matches_direct_reverse_kl_gradient(self):
        teacher = torch.tensor([[[0.0, 1.0, -1.0]]])
        surrogate_student = torch.tensor([[[1.0, 0.0, -1.0]]], requires_grad=True)
        surrogate, _ = compute_opd_loss(
            surrogate_student,
            teacher,
            torch.tensor([[0]]),
            torch.tensor([[1]]),
            distill_top_k=-1,
        )
        surrogate.backward()

        direct_student = surrogate_student.detach().clone().requires_grad_(True)
        student_logps = torch.log_softmax(direct_student, dim=-1)
        teacher_logps = torch.log_softmax(teacher, dim=-1)
        direct_reverse_kl = (student_logps.exp() * (student_logps - teacher_logps)).sum()
        direct_reverse_kl.backward()
        self.assertTrue(torch.allclose(surrogate_student.grad, direct_student.grad, atol=1e-6))

    def test_topk_weight_modes_are_finite(self):
        for weight_mode in ("student_p", "teacher_p", "none"):
            student = torch.tensor([[[0.5, 0.0, -0.5]]], requires_grad=True)
            teacher = torch.tensor([[[-0.5, 0.0, 0.5]]])
            loss, metrics = compute_opd_loss(
                student,
                teacher,
                torch.tensor([[0]]),
                torch.tensor([[1]]),
                distill_top_k=2,
                weight_mode=weight_mode,
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(metrics["topk_overlap_ratio"]))

    def test_none_weight_uses_one_per_candidate_instead_of_one_over_k(self):
        student = torch.tensor([[[2.0, 1.0, 0.0]]], requires_grad=True)
        teacher = torch.tensor([[[0.0, 1.0, 2.0]]])
        loss, _ = compute_opd_loss(
            student,
            teacher,
            torch.tensor([[0]]),
            torch.tensor([[1]]),
            distill_top_k=2,
            weight_mode="none",
        )
        student_logps = torch.log_softmax(student, dim=-1)
        teacher_logps = torch.log_softmax(teacher, dim=-1)
        candidate_ids = student_logps.detach().topk(k=2, dim=-1).indices
        student_selected = student_logps.gather(-1, candidate_ids)
        teacher_selected = teacher_logps.gather(-1, candidate_ids)
        expected = (
            (student_selected.detach() - teacher_selected.detach()) * student_selected
        ).sum()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-7))

    def test_all_topk_candidate_strategies_are_finite(self):
        for strategy in ("only_stu", "only_tch", "intersection", "union"):
            student = torch.tensor([[[2.0, 1.0, 0.0, -1.0]]], requires_grad=True)
            teacher = torch.tensor([[[-1.0, 0.0, 1.0, 2.0]]])
            loss, metrics = compute_opd_loss(
                student,
                teacher,
                torch.tensor([[0]]),
                torch.tensor([[1]]),
                distill_top_k=2,
                top_k_strategy=strategy,
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss), msg=strategy)
            self.assertTrue(torch.isfinite(metrics["reverse_kl"]), msg=strategy)

    def test_union_does_not_double_count_overlapping_candidates(self):
        student = torch.tensor([[[1.0, 0.5, 0.0]]], requires_grad=True)
        teacher = student.detach().clone()
        loss, metrics = compute_opd_loss(
            student,
            teacher,
            torch.tensor([[0]]),
            torch.tensor([[1]]),
            distill_top_k=2,
            top_k_strategy="union",
        )
        loss.backward()
        self.assertAlmostEqual(loss.item(), 0.0, places=7)
        self.assertAlmostEqual(metrics["reverse_kl"].item(), 0.0, places=7)
        self.assertTrue(torch.allclose(student.grad, torch.zeros_like(student.grad), atol=1e-7))

    def test_masked_tokens_have_no_gradient(self):
        student = torch.zeros(1, 2, 3, requires_grad=True)
        teacher = torch.tensor([[[-2.0, 2.0, -2.0], [2.0, -2.0, -2.0]]])
        loss, _ = compute_opd_loss(
            student, teacher, torch.tensor([[1, 0]]), torch.tensor([[1, 0]]), distill_top_k=3
        )
        loss.backward()
        self.assertTrue(torch.allclose(student.grad[0, 1], torch.zeros(3), atol=1e-7))

    def test_empty_completion_mask_is_finite_and_has_zero_gradient(self):
        student = torch.zeros(1, 1, 3, requires_grad=True)
        teacher = torch.tensor([[[-2.0, 2.0, -2.0]]])
        loss, metrics = compute_opd_loss(
            student, teacher, torch.tensor([[1]]), torch.tensor([[0]]), distill_top_k=3
        )
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.allclose(student.grad, torch.zeros_like(student.grad)))
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
