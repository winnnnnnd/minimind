"""Tensor utilities for MiniMind on-policy distillation (OPD)."""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def build_completion_mask(
    completion_ids: Tensor,
    base_mask: Optional[Tensor],
    eos_token_id: Optional[int],
    pad_token_id: Optional[int] = None,
) -> Tensor:
    """Return a boolean mask that keeps valid tokens through the first EOS.

    MiniMind's native ``generate`` fills already-finished rows with repeated EOS
    tokens until every row is done.  Those repeated tokens must not contribute to
    OPD.  ``base_mask`` additionally carries variable-length information from a
    remote rollout engine.
    """
    if completion_ids.ndim != 2:
        raise ValueError(f"completion_ids must be 2D, got shape={tuple(completion_ids.shape)}")
    if base_mask is None:
        mask = torch.ones_like(completion_ids, dtype=torch.bool)
    else:
        if base_mask.shape != completion_ids.shape:
            raise ValueError(
                f"base_mask shape {tuple(base_mask.shape)} does not match completion_ids "
                f"shape {tuple(completion_ids.shape)}"
            )
        mask = base_mask.bool().clone()

    # Do not remove EOS when a tokenizer intentionally aliases PAD and EOS.
    if pad_token_id is not None and pad_token_id != eos_token_id:
        mask &= completion_ids.ne(pad_token_id)

    if eos_token_id is not None:
        eos_hits = completion_ids.eq(eos_token_id) & mask
        eos_seen_before = eos_hits.long().cumsum(dim=1) - eos_hits.long()
        mask &= eos_seen_before.eq(0)

    return mask


def extract_incremental_observation_suffix(
    closed_assistant_ids: Sequence[int],
    observed_context_ids: Sequence[int],
    sampled_action_ids: Sequence[int],
    eos_token_id: Optional[int],
) -> List[int]:
    """Extract only the template/tool suffix following a sampled assistant action.

    A decoded action is not guaranteed to re-encode to the IDs sampled by the
    policy (for example, ``["Ġ3", "6"]`` may canonicalize to ``["Ġ36"]``).
    Both input contexts here are canonical template encodings, so their shared
    prefix is safe for locating the assistant boundary.  The returned suffix
    can then be appended to the original sampled IDs without re-tokenizing or
    replacing them.
    """
    closed_assistant_ids = list(closed_assistant_ids)
    observed_context_ids = list(observed_context_ids)
    sampled_action_ids = list(sampled_action_ids)
    if eos_token_id is None:
        raise ValueError("incremental tool observations require an eos_token_id")
    if observed_context_ids[: len(closed_assistant_ids)] != closed_assistant_ids:
        raise ValueError("closed assistant rendering is not a prefix of tool observation rendering")
    try:
        assistant_eos_index = len(closed_assistant_ids) - 1 - closed_assistant_ids[::-1].index(
            eos_token_id
        )
    except ValueError as exc:
        raise ValueError("closed assistant rendering does not contain an EOS boundary") from exc

    # Native generation includes the sampled EOS but not the template newline
    # following it.  If generation was truncated before EOS, include the
    # template-supplied close token as masked environment context as well.
    suffix_start = (
        assistant_eos_index + 1
        if sampled_action_ids and sampled_action_ids[-1] == eos_token_id
        else assistant_eos_index
    )
    return observed_context_ids[suffix_start:]


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def is_reference_replay_step(step: int, replay_ratio: float) -> bool:
    """Evenly distribute reference batches without random Bernoulli draws."""
    if not 0 < replay_ratio <= 1:
        raise ValueError("replay_ratio must be in (0, 1]")
    if step < 1:
        return False
    epsilon = 1e-12
    return math.floor(step * replay_ratio + epsilon) > math.floor(
        (step - 1) * replay_ratio + epsilon
    )


def compute_reference_kl_loss(
    student_logits: Tensor,
    reference_logits: Tensor,
    token_mask: Tensor,
    student_temperature: float = 1.0,
    reference_temperature: float = 1.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute exact reverse KL from the student to a frozen Base policy.

    This loss is used on a separate replay stream containing ordinary Base
    model conversations.  ``token_mask`` should select only assistant target
    positions; user, system, tool and padding tokens are context rather than
    optimization targets.
    """
    if student_logits.ndim != 3 or reference_logits.ndim != 3:
        raise ValueError("student_logits and reference_logits must both be 3D [batch, time, vocab]")
    if student_logits.shape != reference_logits.shape:
        raise ValueError(
            "student/reference logits must have identical shapes, got "
            f"{tuple(student_logits.shape)} and {tuple(reference_logits.shape)}"
        )
    if token_mask.shape != student_logits.shape[:2]:
        raise ValueError(
            f"token_mask shape {tuple(token_mask.shape)} must match logits time dimensions "
            f"{tuple(student_logits.shape[:2])}"
        )
    if student_temperature <= 0 or reference_temperature <= 0:
        raise ValueError("student and reference temperatures must be positive")

    # Compute full-vocabulary KL in FP32 for numerical stability while keeping
    # gradients connected to a lower-precision student forward pass.
    student_logps = F.log_softmax(student_logits.float() / student_temperature, dim=-1)
    # The reference policy is a fixed target even if a caller accidentally
    # forgets to wrap its forward pass in torch.no_grad().
    reference_logps = F.log_softmax(
        reference_logits.detach().float() / reference_temperature, dim=-1
    )
    student_probs = student_logps.exp()
    per_token_kl = (student_probs * (student_logps - reference_logps)).sum(dim=-1)
    mask = token_mask.bool()
    loss = _masked_mean(per_token_kl, mask)

    student_top1 = student_logps.detach().argmax(dim=-1)
    reference_top1 = reference_logps.argmax(dim=-1)
    metrics = {
        "reference_reverse_kl": loss.detach(),
        "reference_top1_agreement": _masked_mean(
            student_top1.eq(reference_top1).float(), mask
        ).detach(),
        "reference_token_count": mask.sum().detach(),
    }
    return loss, metrics


def _reward_weights(
    student_logps: Tensor,
    teacher_logps: Tensor,
    mode: str,
    candidate_mask: Optional[Tensor] = None,
) -> Tensor:
    """Return detached token weights without renormalizing top-k support.

    OPD uses probabilities from the original full-vocabulary distribution.  A
    softmax over the selected candidates would incorrectly turn a small top-k
    probability mass into one.  ``none`` means one coefficient per candidate,
    not a uniform distribution over candidates.
    """
    if mode == "student_p":
        weights = student_logps.detach().exp()
    elif mode == "teacher_p":
        weights = teacher_logps.detach().exp()
    elif mode == "none":
        weights = torch.ones_like(student_logps)
    else:
        raise ValueError(f"weight_mode must be one of student_p/teacher_p/none, got {mode!r}")
    if candidate_mask is not None:
        weights = weights * candidate_mask.to(device=weights.device, dtype=weights.dtype)
    return weights


def _select_topk_candidates(
    student_logps: Tensor,
    teacher_logps: Tensor,
    k: int,
    strategy: str,
) -> Tuple[Tensor, Tensor]:
    """Select candidate token IDs and a mask for set operations.

    The union is represented by concatenating both top-k lists and masking the
    teacher-side duplicates.  This keeps tensor shapes static while ensuring a
    token contributes only once.
    """
    student_ids = student_logps.detach().topk(k=k, dim=-1).indices
    teacher_ids = teacher_logps.detach().topk(k=k, dim=-1).indices

    if strategy == "only_stu":
        return student_ids, torch.ones_like(student_ids, dtype=torch.bool)
    if strategy == "only_tch":
        return teacher_ids, torch.ones_like(teacher_ids, dtype=torch.bool)

    student_in_teacher = student_ids.unsqueeze(-1).eq(teacher_ids.unsqueeze(-2)).any(dim=-1)
    if strategy == "intersection":
        return student_ids, student_in_teacher
    if strategy == "union":
        teacher_in_student = teacher_ids.unsqueeze(-1).eq(student_ids.unsqueeze(-2)).any(dim=-1)
        candidate_ids = torch.cat([student_ids, teacher_ids], dim=-1)
        candidate_mask = torch.cat(
            [torch.ones_like(student_ids, dtype=torch.bool), ~teacher_in_student], dim=-1
        )
        return candidate_ids, candidate_mask
    raise ValueError(
        "top_k_strategy must be one of only_stu/only_tch/intersection/union, "
        f"got {strategy!r}"
    )


def compute_opd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    completion_ids: Tensor,
    completion_mask: Tensor,
    distill_top_k: int = 16,
    weight_mode: str = "student_p",
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
    top_k_strategy: str = "only_stu",
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute an OPD reverse-KL policy-gradient surrogate.

    Modes are selected by ``distill_top_k``:

    - ``0``: sampled-token K1 estimator;
    - ``> 0``: top-k support selected by ``top_k_strategy``;
    - ``-1``: full-vocabulary support, useful as a low-variance reference.

    The KL-derived token advantages and probability weights are detached.  The
    remaining student log-probability factor supplies the policy gradient.
    """
    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("student_logits and teacher_logits must both be 3D [batch, response, vocab]")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher logits must have identical shapes, got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    if completion_ids.shape != student_logits.shape[:2] or completion_mask.shape != completion_ids.shape:
        raise ValueError("completion_ids and completion_mask must match logits [batch, response] dimensions")
    if distill_top_k < -1:
        raise ValueError(f"distill_top_k must be -1, 0, or a positive integer, got {distill_top_k}")
    if student_temperature <= 0 or teacher_temperature <= 0:
        raise ValueError("student and teacher temperatures must be positive")
    if weight_mode not in {"student_p", "teacher_p", "none"}:
        raise ValueError(f"unsupported weight_mode={weight_mode!r}")
    if top_k_strategy not in {"only_stu", "only_tch", "intersection", "union"}:
        raise ValueError(f"unsupported top_k_strategy={top_k_strategy!r}")

    vocab_size = student_logits.shape[-1]
    student_logps = F.log_softmax(student_logits / student_temperature, dim=-1)
    teacher_logps = F.log_softmax(teacher_logits / teacher_temperature, dim=-1)
    mask = completion_mask.bool()

    student_top1 = student_logps.detach().argmax(dim=-1)
    teacher_top1 = teacher_logps.detach().argmax(dim=-1)
    top1_agreement = _masked_mean(student_top1.eq(teacher_top1).float(), mask)

    if distill_top_k == 0:
        if completion_ids.numel() and (
            completion_ids.min().item() < 0 or completion_ids.max().item() >= vocab_size
        ):
            raise ValueError("completion_ids contain a token outside the student/teacher vocabulary")
        sampled_ids = completion_ids.to(student_logits.device).long().unsqueeze(-1)
        student_selected = student_logps.gather(dim=-1, index=sampled_ids).squeeze(-1)
        teacher_selected = teacher_logps.gather(dim=-1, index=sampled_ids).squeeze(-1)
        log_ratio = student_selected.detach() - teacher_selected.detach()
        loss = _masked_mean(log_ratio * student_selected, mask)
        reverse_kl = _masked_mean(log_ratio, mask)
        overlap_ratio = student_logits.new_zeros(())
        overlap_token_advantage = student_logits.new_zeros(())
    else:
        k = vocab_size if distill_top_k == -1 else min(distill_top_k, vocab_size)
        if distill_top_k == -1:
            candidate_ids = torch.arange(vocab_size, device=student_logits.device)
            candidate_ids = candidate_ids.view(1, 1, -1).expand(*student_logps.shape[:2], -1)
            candidate_mask = torch.ones_like(candidate_ids, dtype=torch.bool)
        else:
            candidate_ids, candidate_mask = _select_topk_candidates(
                student_logps, teacher_logps, k, top_k_strategy
            )
        student_selected = student_logps.gather(dim=-1, index=candidate_ids)
        teacher_selected = teacher_logps.gather(dim=-1, index=candidate_ids)
        weights = _reward_weights(
            student_selected, teacher_selected, weight_mode, candidate_mask=candidate_mask
        )
        log_ratio = student_selected.detach() - teacher_selected.detach()
        token_advantages = weights * log_ratio
        per_state_loss = (token_advantages * student_selected).sum(dim=-1)
        per_state_reverse_kl = (weights * log_ratio).sum(dim=-1)
        supported_state_mask = mask & candidate_mask.any(dim=-1)
        loss = _masked_mean(per_state_loss, supported_state_mask)
        reverse_kl = _masked_mean(per_state_reverse_kl, supported_state_mask)

        if distill_top_k == -1:
            overlap_ratio = student_logits.new_ones(())
            overlap_token_advantage = _masked_mean(-per_state_reverse_kl, supported_state_mask)
        else:
            student_topk_ids = student_logps.detach().topk(k=k, dim=-1).indices
            teacher_topk_ids = teacher_logps.detach().topk(k=k, dim=-1).indices
            overlap = student_topk_ids.unsqueeze(-1).eq(teacher_topk_ids.unsqueeze(-2)).any(dim=-1)
            overlap_ratio = _masked_mean(overlap.float().mean(dim=-1), mask)
            overlap_student_logps = student_logps.gather(dim=-1, index=student_topk_ids)
            overlap_teacher_logps = teacher_logps.gather(dim=-1, index=student_topk_ids)
            overlap_weights = _reward_weights(
                overlap_student_logps,
                overlap_teacher_logps,
                weight_mode,
                candidate_mask=overlap,
            )
            overlap_denom = overlap_weights.sum(dim=-1)
            per_state_overlap_advantage = (
                overlap_weights * (overlap_teacher_logps.detach() - overlap_student_logps.detach())
            ).sum(dim=-1) / overlap_denom.clamp(min=torch.finfo(overlap_weights.dtype).eps)
            valid_overlap_mask = mask & overlap_denom.gt(0)
            overlap_token_advantage = _masked_mean(per_state_overlap_advantage, valid_overlap_mask)

    metrics = {
        "reverse_kl": reverse_kl.detach(),
        "teacher_token_advantage": (-reverse_kl).detach(),
        "topk_overlap_ratio": overlap_ratio.detach(),
        "overlap_token_advantage": overlap_token_advantage.detach(),
        "teacher_top1_agreement": top1_agreement.detach(),
    }
    return loss, metrics
