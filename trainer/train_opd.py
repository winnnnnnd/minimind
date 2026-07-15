"""On-policy distillation for MiniMind Agent/tool-use trajectories.

The student samples every assistant action. Tool results are supplied by the
environment and masked out of the OPD objective. The teacher is evaluated on
the student's complete trajectory; it never generates replacement answers.
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

__package__ = "trainer"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from dataset.lm_dataset import AgentRLDataset, ReferenceReplayDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.opd_utils import (
    build_completion_mask,
    compute_opd_loss,
    compute_reference_kl_loss,
    extract_incremental_observation_suffix,
    is_reference_replay_step,
)
from trainer.rollout_engine import create_rollout_engine
from trainer.tool_utils import validate_gt_in_text
from trainer.train_agent import execute_tool, parse_tool_calls
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    init_distributed_mode,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)

warnings.filterwarnings("ignore")


class TrajectoryAlignmentError(RuntimeError):
    """A sampled action cannot be losslessly extended with a rendered tool observation."""


def _alignment_error(reason, rendered_ids, sampled_ids, action_text, turn):
    common_length = min(len(rendered_ids), len(sampled_ids))
    mismatch_index = next(
        (index for index in range(common_length) if rendered_ids[index] != sampled_ids[index]),
        common_length,
    )
    window_start = max(0, mismatch_index - 12)
    window_end = mismatch_index + 20
    rendered_window = rendered_ids[window_start:window_end]
    sampled_window = sampled_ids[window_start:window_end]
    return TrajectoryAlignmentError(
        f"{reason}; turn={turn + 1}, first_mismatch={mismatch_index}, "
        f"rendered_len={len(rendered_ids)}, sampled_len={len(sampled_ids)}, "
        f"rendered_tokens={rendered_window!r}, sampled_tokens={sampled_window!r}, "
        f"last_action={action_text[-240:]!r}"
    )


def resolve_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_path(value):
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def validate_args(parsed_args):
    if parsed_args.distill_top_k < -1:
        raise ValueError("--distill_top_k must be -1 (full), 0 (sampled K1), or a positive integer")
    if parsed_args.rollout_temperature <= 0:
        raise ValueError("--rollout_temperature must be positive")
    if parsed_args.student_temperature <= 0 or parsed_args.teacher_temperature <= 0:
        raise ValueError("student and teacher temperatures must be positive")
    if not 0 < parsed_args.rollout_top_p <= 1:
        raise ValueError("--rollout_top_p must be in (0, 1]")
    if parsed_args.rollout_top_k < 0:
        raise ValueError("--rollout_top_k must be zero (disabled) or positive")
    if parsed_args.num_generations < 1 or parsed_args.max_gen_len < 1:
        raise ValueError("--num_generations and --max_gen_len must be positive")
    if parsed_args.rollout_alignment_retries < 0:
        raise ValueError("--rollout_alignment_retries must be zero or positive")
    if parsed_args.max_turns < 1 or parsed_args.max_total_len < 2:
        raise ValueError("--max_turns must be positive and --max_total_len must be at least two")
    if parsed_args.accumulation_steps < 1:
        raise ValueError("--accumulation_steps must be positive")
    if parsed_args.log_interval < 1 or parsed_args.save_interval < 1 or parsed_args.debug_interval < 1:
        raise ValueError("log/save/debug intervals must be positive")
    if parsed_args.max_train_steps < 0:
        raise ValueError("--max_train_steps must be zero or positive")
    if parsed_args.lambda_ref < 0:
        raise ValueError("--lambda_ref must be zero or positive")
    if not 0 < parsed_args.reference_replay_ratio <= 1:
        raise ValueError("--reference_replay_ratio must be in (0, 1]")
    if parsed_args.reference_batch_size < 1 or parsed_args.reference_max_len < 2:
        raise ValueError("--reference_batch_size must be positive and --reference_max_len at least two")
    if parsed_args.reference_max_samples < 0 or parsed_args.reference_sample_stride < 1:
        raise ValueError("--reference_max_samples must be non-negative and --reference_sample_stride positive")
    if parsed_args.reference_temperature <= 0:
        raise ValueError("--reference_temperature must be positive")
    if not 0 <= parsed_args.metrics_ema_decay < 1:
        raise ValueError("--metrics_ema_decay must be in [0, 1)")
    if not 0 <= parsed_args.thinking_ratio <= 1:
        raise ValueError("--thinking_ratio must be in [0, 1]")
    if parsed_args.rollout_top_p < 1.0 or parsed_args.rollout_top_k > 0:
        Logger(
            "[OPD WARNING] top-p/top-k truncation changes the behavior distribution; "
            "top_p=1 and top_k=0 are recommended for the strict OPD baseline"
        )
    if parsed_args.save_interval % parsed_args.accumulation_steps != 0:
        Logger(
            "[OPD WARNING] --save_interval is not divisible by --accumulation_steps; "
            "periodic checkpoints are only written on optimizer-update boundaries"
        )


def distributed_mean(value):
    result = value.detach().float().clone()
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= dist.get_world_size()
    return result.item()


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_transformers_state_dict(model_dir):
    single_safetensors = model_dir / "model.safetensors"
    safetensors_index = model_dir / "model.safetensors.index.json"
    pytorch_bin = model_dir / "pytorch_model.bin"

    if single_safetensors.exists() or safetensors_index.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Loading a Transformers safetensors model requires safetensors") from exc
        if single_safetensors.exists():
            return load_file(str(single_safetensors), device="cpu")
        index = json.loads(safetensors_index.read_text(encoding="utf-8"))
        state_dict = {}
        for shard_name in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(str(model_dir / shard_name), device="cpu"))
        return state_dict
    if pytorch_bin.exists():
        return _torch_load(pytorch_bin)
    raise FileNotFoundError(
        f"No model.safetensors, model.safetensors.index.json, or pytorch_model.bin in {model_dir}"
    )


def load_state_dict(source):
    source = resolve_path(source)
    if source.is_dir():
        state_dict = _load_transformers_state_dict(source)
    elif source.is_file():
        state_dict = _torch_load(source)
    else:
        raise FileNotFoundError(f"Model source does not exist: {source}")

    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Invalid or empty state dict from {source}")
    for prefix in ("module.", "_orig_mod."):
        if all(key.startswith(prefix) for key in state_dict):
            state_dict = {key[len(prefix) :]: value for key, value in state_dict.items()}
    # Safetensors can omit one side of a tied embedding/lm_head pair.
    if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    if "model.embed_tokens.weight" not in state_dict and "lm_head.weight" in state_dict:
        state_dict["model.embed_tokens.weight"] = state_dict["lm_head.weight"]
    return state_dict, source


def validate_directory_config(source, config, label):
    source = resolve_path(source)
    config_path = source / "config.json" if source.is_dir() else None
    if not config_path or not config_path.exists():
        return
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
    }
    mismatches = {
        key: (saved.get(key), value)
        for key, value in expected.items()
        if saved.get(key) is not None and saved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} config does not match CLI architecture: {mismatches}")


def load_native_model(config, source, device, label, teacher_dtype=None):
    validate_directory_config(source, config, label)
    state_dict, resolved_source = load_state_dict(source)
    model_instance = MiniMindForCausalLM(config)
    model_instance.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()
    if teacher_dtype is None:
        model_instance = model_instance.to(device)
    else:
        model_instance = model_instance.to(device=device, dtype=teacher_dtype)
    Logger(f"Loaded {label} strictly from {resolved_source}")
    return model_instance


def load_and_validate_tokenizer(student_path, teacher_path, expected_vocab_size):
    student_path = resolve_path(student_path)
    teacher_path = resolve_path(teacher_path)
    student_tokenizer = AutoTokenizer.from_pretrained(str(student_path), trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(str(teacher_path), trust_remote_code=True)
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Student and teacher tokenizers do not have the same token-to-ID vocabulary")
    special_names = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    special_mismatches = {
        name: (getattr(student_tokenizer, name), getattr(teacher_tokenizer, name))
        for name in special_names
        if getattr(student_tokenizer, name) != getattr(teacher_tokenizer, name)
    }
    if special_mismatches:
        raise ValueError(f"Student and teacher special token IDs differ: {special_mismatches}")
    if len(student_tokenizer) != expected_vocab_size:
        raise ValueError(
            f"Tokenizer size {len(student_tokenizer)} does not match model vocab {expected_vocab_size}"
        )
    if student_tokenizer.pad_token_id is None or student_tokenizer.eos_token_id is None:
        raise ValueError("OPD requires tokenizer.pad_token_id and tokenizer.eos_token_id")
    Logger(f"Tokenizer compatibility check passed: {len(student_tokenizer)} identical token IDs")
    return student_tokenizer


def _tool_result_message(call):
    name, raw_args = call.get("name", ""), call.get("arguments", {})
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            raw_args = {}
    result = execute_tool(name, raw_args)
    if result is None:
        result = {"error": f"tool execution failed: {name}"}
    return json.dumps(result, ensure_ascii=False)[:2048], name


def rollout_agent_trajectory(rollout_engine, messages, tools):
    """Sample one coherent multi-turn student trajectory with action masks."""
    messages = [dict(message) for message in messages]
    open_thinking = random.random() < args.thinking_ratio
    prompt_context = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        open_thinking=open_thinking,
    )
    prompt_ids = tokenizer(prompt_context, add_special_tokens=False).input_ids
    response_ids = []
    response_mask = []
    turn_outputs = []
    called_tools = []
    unfinished = False

    for turn in range(args.max_turns):
        # Keep the policy's sampled IDs authoritative.  Re-rendering the full
        # conversation can canonicalize a valid alternative BPE segmentation
        # and would silently turn an on-policy trajectory into another token
        # sequence.
        context_ids = prompt_ids + response_ids
        remaining = args.max_total_len - len(context_ids)
        if remaining <= 0:
            raise ValueError(
                f"Rendered prompt already has {len(context_ids)} tokens, exceeding --max_total_len="
                f"{args.max_total_len}"
            )

        input_ids = torch.tensor([context_ids], dtype=torch.long, device=args.device)
        attention_mask = torch.ones_like(input_ids)
        rollout_result = rollout_engine.rollout(
            prompt_ids=input_ids,
            attention_mask=attention_mask,
            num_generations=1,
            max_new_tokens=min(args.max_gen_len, remaining),
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            calculate_logps=False,
        )
        completion_ids = rollout_result.completion_ids.to(args.device)
        valid_mask = build_completion_mask(
            completion_ids,
            rollout_result.completion_mask.to(args.device),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )[0]
        action_ids = completion_ids[0][valid_mask].tolist()
        if not action_ids:
            raise RuntimeError("Student rollout returned no valid action tokens")
        action_text = tokenizer.decode(action_ids, skip_special_tokens=True)
        turn_outputs.append(action_text)
        response_ids.extend(action_ids)
        response_mask.extend([1] * len(action_ids))

        calls = parse_tool_calls(action_text)
        if not calls:
            break
        for call in calls:
            called_tools.append(call.get("name", ""))
        if turn == args.max_turns - 1:
            unfinished = True
            break

        messages.append({"role": "assistant", "content": action_text})
        closed_assistant_context = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
            open_thinking=open_thinking,
        )
        closed_assistant_ids = tokenizer(
            closed_assistant_context, add_special_tokens=False
        ).input_ids
        for call in calls:
            result_text, _ = _tool_result_message(call)
            messages.append({"role": "tool", "content": result_text})

        observed_context = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
            open_thinking=open_thinking,
        )
        observed_ids = tokenizer(observed_context, add_special_tokens=False).input_ids
        if observed_ids[: len(closed_assistant_ids)] != closed_assistant_ids:
            raise _alignment_error(
                "Tool observation rendering changed the canonical closed assistant prefix",
                observed_ids[: len(closed_assistant_ids)],
                closed_assistant_ids,
                action_text,
                turn,
            )
        observation_delta = extract_incremental_observation_suffix(
            closed_assistant_ids,
            observed_ids,
            action_ids,
            tokenizer.eos_token_id,
        )
        response_ids.extend(observation_delta)
        response_mask.extend([0] * len(observation_delta))

    final_output = turn_outputs[-1] if turn_outputs else ""
    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "response_mask": response_mask,
        "final_output": final_output,
        "turn_outputs": turn_outputs,
        "called_tools": called_tools,
        "unfinished": unfinished,
    }


def rollout_agent_batch(rollout_engine, messages_batch, tools_batch, gt_batch):
    trajectories = []
    for batch_index, (messages, tools, gt) in enumerate(
        zip(messages_batch, tools_batch, gt_batch), start=1
    ):
        for generation_index in range(args.num_generations):
            last_error = None
            for attempt in range(args.rollout_alignment_retries + 1):
                try:
                    trajectory = rollout_agent_trajectory(rollout_engine, messages, tools)
                except TrajectoryAlignmentError as exc:
                    last_error = exc
                    Logger(
                        f"[OPD ALIGNMENT] batch_row={batch_index}, "
                        f"generation={generation_index + 1}, "
                        f"attempt={attempt + 1}/{args.rollout_alignment_retries + 1}: {exc}"
                    )
                    continue
                trajectory["gt"] = list(gt)
                trajectory["alignment_retries"] = attempt
                trajectories.append(trajectory)
                break
            else:
                raise RuntimeError(
                    "Unable to sample a tokenizer-aligned multi-turn trajectory after "
                    f"{args.rollout_alignment_retries + 1} attempts"
                ) from last_error
    return trajectories


def pack_trajectories(trajectories):
    samples = []
    for trajectory in trajectories:
        ids = trajectory["prompt_ids"] + trajectory["response_ids"]
        action_mask = [0] * len(trajectory["prompt_ids"]) + trajectory["response_mask"]
        if len(ids) > args.max_total_len:
            raise RuntimeError(
                f"Trajectory length {len(ids)} exceeds --max_total_len={args.max_total_len}; "
                "rollout length accounting is inconsistent"
            )
        if len(ids) < 2 or not any(action_mask[1:]):
            raise RuntimeError("A packed OPD trajectory must contain at least one predicted action token")
        samples.append((ids, action_mask))

    max_len = max(len(ids) for ids, _ in samples)
    input_ids = torch.tensor(
        [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _ in samples],
        dtype=torch.long,
        device=args.device,
    )
    attention_mask = torch.tensor(
        [[1] * len(ids) + [0] * (max_len - len(ids)) for ids, _ in samples],
        dtype=torch.long,
        device=args.device,
    )
    full_action_mask = torch.tensor(
        [mask + [0] * (max_len - len(mask)) for _, mask in samples],
        dtype=torch.bool,
        device=args.device,
    )
    return input_ids, attention_mask, input_ids[:, 1:], full_action_mask[:, 1:]


def collate_reference_batch(batch):
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full(
        (len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    assistant_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for row, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[row, :length] = item["input_ids"]
        attention_mask[row, :length] = 1
        assistant_mask[row, :length] = item["assistant_mask"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "assistant_mask": assistant_mask,
    }


def trajectory_metrics(trajectories):
    successes = []
    tool_call_rows = []
    math_call_rows = []
    unfinished_rows = []
    response_lengths = []
    alignment_retries = []
    for trajectory in trajectories:
        gt = trajectory["gt"]
        verified = validate_gt_in_text(trajectory["final_output"], gt) if gt else set()
        successes.append(float(bool(gt) and len(verified) == len(gt) and not trajectory["unfinished"]))
        tool_call_rows.append(float(bool(trajectory["called_tools"])))
        math_call_rows.append(float("calculate_math" in trajectory["called_tools"]))
        unfinished_rows.append(float(trajectory["unfinished"]))
        response_lengths.append(float(sum(trajectory["response_mask"])))
        alignment_retries.append(float(trajectory.get("alignment_retries", 0)))
    device = args.device
    return {
        "verified_success_rate": torch.tensor(successes, device=device).mean(),
        "tool_call_rate": torch.tensor(tool_call_rows, device=device).mean(),
        "math_call_rate": torch.tensor(math_call_rows, device=device).mean(),
        "unfinished_rate": torch.tensor(unfinished_rows, device=device).mean(),
        "avg_action_tokens": torch.tensor(response_lengths, device=device).mean(),
        "alignment_retry_rate": torch.tensor(
            [float(retries > 0) for retries in alignment_retries], device=device
        ).mean(),
        "avg_alignment_retries": torch.tensor(alignment_retries, device=device).mean(),
    }


def save_student(lm_config, epoch, step, wandb):
    if not is_main_process():
        return
    model.eval()
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    state_dict = {key: value.detach().half().cpu() for key, value in raw_model.state_dict().items()}
    save_dir = resolve_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_moe" if lm_config.use_moe else ""
    latest_path = save_dir / f"{args.save_weight}_{lm_config.hidden_size}{suffix}.pth"
    temporary_path = latest_path.with_suffix(latest_path.suffix + ".tmp")
    torch.save(state_dict, temporary_path)
    os.replace(temporary_path, latest_path)
    if args.keep_step_checkpoints:
        step_path = save_dir / f"{args.save_weight}_{lm_config.hidden_size}{suffix}_step{step:06d}.pth"
        torch.save(state_dict, step_path)
    lm_checkpoint(
        lm_config,
        weight=args.save_weight,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        wandb=wandb,
        save_dir=str(resolve_path(args.checkpoint_dir)),
        scheduler=scheduler,
        scaler=scaler,
    )
    Logger(f"Saved student checkpoint to {latest_path}")
    model.train()
    del state_dict


def optimizer_step():
    scaler.unscale_(optimizer)
    if args.grad_clip > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    else:
        grad_norm = next(model.parameters()).new_zeros(())
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm.detach()


def update_ema(ema_state, name, value, decay):
    previous = ema_state.get(name)
    ema_state[name] = value if previous is None else decay * previous + (1 - decay) * value
    return ema_state[name]


def opd_train_epoch(
    epoch,
    loader,
    iters,
    rollout_engine,
    teacher_model,
    reference_model,
    reference_loader,
    lm_config,
    start_step=0,
    wandb=None,
):
    last_step = start_step
    accumulated = 0
    optimizer_updates = start_step // args.accumulation_steps
    last_grad_norm = None
    metrics_ema = {}
    epoch_started_at = time.time()
    teacher_model.eval().requires_grad_(False)
    if reference_model is not None:
        reference_model.eval().requires_grad_(False)
    reference_iterator = iter(reference_loader) if reference_loader is not None else None
    if reference_iterator is not None and start_step > 0:
        # Restore the fixed replay-stream position when resuming an OPD epoch.
        completed_replays = math.floor(start_step * args.reference_replay_ratio + 1e-12)
        for _ in range(completed_replays % len(reference_loader)):
            next(reference_iterator)
    model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        if args.max_train_steps > 0 and step > args.max_train_steps:
            break
        step_started_at = time.time()
        last_step = step

        model.eval()
        with torch.no_grad():
            trajectories = rollout_agent_batch(
                rollout_engine, batch["messages"], batch["tools"], batch["gt"]
            )
        model.train()
        input_ids, attention_mask, completion_ids, completion_mask = pack_trajectories(trajectories)

        with torch.no_grad(), autocast_ctx:
            teacher_output = teacher_model(input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_output.logits[:, :-1, :]
        with autocast_ctx:
            student_output = model(input_ids, attention_mask=attention_mask)
            student_logits = student_output.logits[:, :-1, :]
            opd_loss, opd_metrics = compute_opd_loss(
                student_logits,
                teacher_logits,
                completion_ids,
                completion_mask,
                distill_top_k=args.distill_top_k,
                weight_mode=args.weight_mode,
                student_temperature=args.student_temperature,
                teacher_temperature=args.teacher_temperature,
                top_k_strategy=args.top_k_strategy,
            )
            aux_loss = student_output.aux_loss if lm_config.use_moe else opd_loss.new_zeros(())
            loss = (opd_loss + aux_loss) / args.accumulation_steps
        scaler.scale(loss).backward()

        reference_replayed = (
            reference_iterator is not None
            and is_reference_replay_step(step, args.reference_replay_ratio)
        )
        reference_loss = opd_loss.detach().new_zeros(())
        reference_metrics = {
            "reference_reverse_kl": reference_loss,
            "reference_top1_agreement": reference_loss,
            "reference_token_count": reference_loss,
        }

        # The OPD graph contains a long sampled trajectory. Free it before the
        # Base replay forward to keep three-model MPS memory usage tractable.
        del student_output, teacher_output, student_logits, teacher_logits, loss

        if reference_replayed:
            try:
                reference_batch = next(reference_iterator)
            except StopIteration:
                reference_iterator = iter(reference_loader)
                reference_batch = next(reference_iterator)
            reference_input_ids = reference_batch["input_ids"].to(args.device)
            reference_attention_mask = reference_batch["attention_mask"].to(args.device)
            reference_token_mask = (
                reference_batch["assistant_mask"][:, 1:].to(args.device)
                & reference_attention_mask[:, 1:].bool()
            )
            with torch.no_grad(), autocast_ctx:
                reference_output = reference_model(
                    reference_input_ids, attention_mask=reference_attention_mask
                )
                reference_logits = reference_output.logits[:, :-1, :]
            with autocast_ctx:
                replay_student_output = model(
                    reference_input_ids, attention_mask=reference_attention_mask
                )
                replay_student_logits = replay_student_output.logits[:, :-1, :]
                reference_loss, reference_metrics = compute_reference_kl_loss(
                    replay_student_logits,
                    reference_logits,
                    reference_token_mask,
                    student_temperature=args.student_temperature,
                    reference_temperature=args.reference_temperature,
                )
                # Replay only occurs on a fraction of steps. The inverse-rate
                # factor makes its long-run expected coefficient lambda_ref.
                scaled_reference_loss = (
                    args.lambda_ref / args.reference_replay_ratio
                ) * reference_loss / args.accumulation_steps
            scaler.scale(scaled_reference_loss).backward()
            del reference_batch, reference_input_ids, reference_attention_mask
            del reference_token_mask, reference_output, reference_logits
            del replay_student_output, replay_student_logits, scaled_reference_loss

        accumulated += 1
        reached_run_limit = args.max_train_steps > 0 and step >= args.max_train_steps
        should_step = accumulated >= args.accumulation_steps or step == iters or reached_run_limit
        if should_step:
            last_grad_norm = optimizer_step()
            optimizer_updates += 1
            accumulated = 0

        rollout_metrics = trajectory_metrics(trajectories)
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            first = trajectories[0]
            Logger(f"[OPD DEBUG] step={step}, gt={first['gt']!r}, tools={first['called_tools']!r}")
            for turn_index, turn_output in enumerate(first["turn_outputs"], start=1):
                Logger(f"[OPD DEBUG] turn={turn_index}: {turn_output!r}")

        if step % args.log_interval == 0 or step == iters or reached_run_limit:
            step_seconds = time.time() - step_started_at
            elapsed_seconds = time.time() - epoch_started_at
            completed_this_epoch = max(step - start_step, 1)
            eta_minutes = (
                elapsed_seconds / completed_this_epoch * max(iters - step, 0) / 60
            )
            weighted_reference_loss = (
                args.lambda_ref / args.reference_replay_ratio * reference_loss.detach()
                if reference_replayed
                else reference_loss
            )
            values = {
                "opd_loss": distributed_mean(opd_loss),
                "reverse_kl": distributed_mean(opd_metrics["reverse_kl"]),
                "teacher_token_advantage": distributed_mean(opd_metrics["teacher_token_advantage"]),
                "topk_overlap_ratio": distributed_mean(opd_metrics["topk_overlap_ratio"]),
                "overlap_token_advantage": distributed_mean(opd_metrics["overlap_token_advantage"]),
                "teacher_top1_agreement": distributed_mean(opd_metrics["teacher_top1_agreement"]),
                "verified_success_rate": distributed_mean(rollout_metrics["verified_success_rate"]),
                "tool_call_rate": distributed_mean(rollout_metrics["tool_call_rate"]),
                "math_call_rate": distributed_mean(rollout_metrics["math_call_rate"]),
                "unfinished_rate": distributed_mean(rollout_metrics["unfinished_rate"]),
                "avg_action_tokens": distributed_mean(rollout_metrics["avg_action_tokens"]),
                "alignment_retry_rate": distributed_mean(
                    rollout_metrics["alignment_retry_rate"]
                ),
                "avg_alignment_retries": distributed_mean(
                    rollout_metrics["avg_alignment_retries"]
                ),
                "aux_loss": distributed_mean(aux_loss),
                "reference_kl": distributed_mean(reference_metrics["reference_reverse_kl"]),
                "reference_top1_agreement": distributed_mean(
                    reference_metrics["reference_top1_agreement"]
                ),
                "reference_token_count": distributed_mean(
                    reference_metrics["reference_token_count"]
                ),
                "reference_replayed": float(reference_replayed),
                "lambda_ref": args.lambda_ref,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "weighted_reference_loss": distributed_mean(weighted_reference_loss),
                "objective_loss": distributed_mean(
                    opd_loss.detach() + aux_loss.detach() + weighted_reference_loss
                ),
                "step_seconds": step_seconds,
                "eta_minutes": eta_minutes,
            }
            values["opd_loss_ema"] = update_ema(
                metrics_ema, "opd_loss", values["opd_loss"], args.metrics_ema_decay
            )
            values["reverse_kl_ema"] = update_ema(
                metrics_ema, "reverse_kl", values["reverse_kl"], args.metrics_ema_decay
            )
            values["verified_success_rate_ema"] = update_ema(
                metrics_ema,
                "verified_success_rate",
                values["verified_success_rate"],
                args.metrics_ema_decay,
            )
            values["math_call_rate_ema"] = update_ema(
                metrics_ema, "math_call_rate", values["math_call_rate"], args.metrics_ema_decay
            )
            values["alignment_retry_rate_ema"] = update_ema(
                metrics_ema,
                "alignment_retry_rate",
                values["alignment_retry_rate"],
                args.metrics_ema_decay,
            )
            if reference_replayed:
                values["reference_kl_ema"] = update_ema(
                    metrics_ema,
                    "reference_kl",
                    values["reference_kl"],
                    args.metrics_ema_decay,
                )
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"OPD Loss: {values['opd_loss']:.4f}, Reverse KL: {values['reverse_kl']:.4f}, "
                f"TopK Overlap: {values['topk_overlap_ratio']:.4f}, "
                f"Top1 Agree: {values['teacher_top1_agreement']:.4f}, "
                f"Ref KL: {values['reference_kl']:.4f}"
                f"{'*' if reference_replayed else '-'}, "
                f"Verified Success: {values['verified_success_rate']:.3f}, "
                f"Tool Call: {values['tool_call_rate']:.3f}, "
                f"LR: {values['learning_rate']:.8f}"
            )
            if wandb and is_main_process():
                visual_values = {
                    "step/data_step": step,
                    "step/optimizer_step": optimizer_updates,
                    "loss/opd": values["opd_loss"],
                    "loss/opd_ema": values["opd_loss_ema"],
                    "loss/aux": values["aux_loss"],
                    "loss/reference_weighted": values["weighted_reference_loss"],
                    "loss/objective": values["objective_loss"],
                    "distill/reverse_kl": values["reverse_kl"],
                    "distill/reverse_kl_ema": values["reverse_kl_ema"],
                    "distill/teacher_token_advantage": values["teacher_token_advantage"],
                    "distill/topk_overlap_ratio": values["topk_overlap_ratio"],
                    "distill/overlap_token_advantage": values["overlap_token_advantage"],
                    "distill/teacher_top1_agreement": values["teacher_top1_agreement"],
                    "rollout/verified_success_rate": values["verified_success_rate"],
                    "rollout/verified_success_rate_ema": values["verified_success_rate_ema"],
                    "rollout/tool_call_rate": values["tool_call_rate"],
                    "rollout/math_call_rate": values["math_call_rate"],
                    "rollout/math_call_rate_ema": values["math_call_rate_ema"],
                    "rollout/unfinished_rate": values["unfinished_rate"],
                    "rollout/avg_action_tokens": values["avg_action_tokens"],
                    "rollout/alignment_retry_rate": values["alignment_retry_rate"],
                    "rollout/alignment_retry_rate_ema": values[
                        "alignment_retry_rate_ema"
                    ],
                    "rollout/avg_alignment_retries": values["avg_alignment_retries"],
                    "reference/replayed": values["reference_replayed"],
                    "optimizer/learning_rate": values["learning_rate"],
                    "optimizer/updated": float(should_step),
                    "system/step_seconds": values["step_seconds"],
                    "system/eta_minutes": values["eta_minutes"],
                }
                if last_grad_norm is not None and should_step:
                    # DDP gradients are synchronized before clipping, so rank-0
                    # can log its local norm without an extra collective here.
                    visual_values["optimizer/grad_norm"] = last_grad_norm.float().item()
                # Keep Base curves sparse: logging a synthetic zero on the 75%
                # non-replay steps would create a misleading saw-tooth plot.
                if reference_replayed:
                    visual_values.update(
                        {
                            "reference/reverse_kl": values["reference_kl"],
                            "reference/reverse_kl_ema": values["reference_kl_ema"],
                            "reference/top1_agreement": values[
                                "reference_top1_agreement"
                            ],
                            "reference/token_count": values["reference_token_count"],
                        }
                    )
                wandb.log(visual_values)

        save_due = step % args.save_interval == 0 or step == iters or reached_run_limit
        if save_due and should_step:
            save_student(lm_config, epoch, step, wandb)

        del input_ids, attention_mask, completion_ids, completion_mask
        del opd_loss, opd_metrics, rollout_metrics, aux_loss, trajectories
        del reference_loss, reference_metrics
        if reached_run_limit:
            break

    if accumulated:
        optimizer_step()
    return last_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Agent OPD (on-policy distillation)")
    parser.add_argument("--student_model", type=str, default="minimind-3", help="学生初始权重：HF目录或.pth")
    parser.add_argument(
        "--teacher_model", type=str, default="model_files/agent_768.pth", help="Agent教师权重：HF目录或.pth"
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        default="minimind-3",
        help="冻结Base reference权重；应与学生初始化权重一致",
    )
    parser.add_argument("--student_tokenizer", type=str, default="minimind-3", help="学生tokenizer路径")
    parser.add_argument("--teacher_tokenizer", type=str, default="model", help="教师tokenizer路径")
    parser.add_argument("--data_path", type=str, default="dataset/agent_rl_math.jsonl", help="Agent prompt JSONL")
    parser.add_argument(
        "--reference_data_path",
        type=str,
        default="dataset/sft_t2t_mini.jsonl",
        help="Base KL通用对话回放JSONL",
    )
    parser.add_argument("--save_dir", type=str, default="out/opd_agent", help="模型权重保存目录")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/opd_agent", help="恢复断点保存目录")
    parser.add_argument("--save_weight", default="opd_agent", type=str, help="保存权重前缀")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="每批prompt数")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0/mps")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="CUDA混合精度；MPS下用于冻结教师权重精度",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载进程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument(
        "--lambda_ref", type=float, default=0.10, help="Base reference KL的长期平均权重；0为关闭"
    )
    parser.add_argument(
        "--reference_replay_ratio",
        type=float,
        default=0.25,
        help="执行Base回放的OPD batch比例；权重会按比例校正",
    )
    parser.add_argument("--reference_batch_size", type=int, default=1, help="Base回放batch size")
    parser.add_argument("--reference_max_len", type=int, default=512, help="Base回放最大序列长度")
    parser.add_argument(
        "--reference_max_samples",
        type=int,
        default=10000,
        help="从大型通用数据文件读取的固定回放池大小；0表示全部读取",
    )
    parser.add_argument(
        "--reference_sample_stride",
        type=int,
        default=10,
        help="通用JSONL按固定行间隔抽取，避免只取文件头部同质数据",
    )
    parser.add_argument("--reference_temperature", type=float, default=1.0, help="Base reference logits温度")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument(
        "--metrics_ema_decay",
        type=float,
        default=0.95,
        help="SwanLab平滑指标的EMA衰减系数",
    )
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--keep_step_checkpoints", action="store_true", help="额外保留每个保存点的独立权重")
    parser.add_argument("--max_gen_len", default=256, type=int, help="每轮学生生成最大长度")
    parser.add_argument("--max_total_len", default=2048, type=int, help="含工具观察的完整轨迹最大长度")
    parser.add_argument("--max_turns", default=3, type=int, help="每条轨迹最大assistant轮数")
    parser.add_argument("--max_train_steps", default=0, type=int, help="每轮最多处理多少批；0表示不限制")
    parser.add_argument("--num_generations", type=int, default=1, help="每个prompt的学生轨迹数")
    parser.add_argument("--rollout_temperature", type=float, default=1.0, help="学生在线采样温度")
    parser.add_argument("--rollout_top_p", type=float, default=1.0, help="学生在线采样top-p")
    parser.add_argument("--rollout_top_k", type=int, default=0, help="学生在线采样top-k；0为关闭")
    parser.add_argument(
        "--rollout_alignment_retries",
        type=int,
        default=3,
        help="多轮轨迹decode/re-encode不完全对齐时的重采样次数；不会放宽严格校验",
    )
    parser.add_argument("--student_temperature", type=float, default=1.0, help="OPD学生logits温度")
    parser.add_argument("--teacher_temperature", type=float, default=1.0, help="OPD教师logits温度")
    parser.add_argument(
        "--distill_top_k", type=int, default=16, help="候选token宽度；16=推荐基线，0=sampled K1，-1=全词表"
    )
    parser.add_argument(
        "--top_k_strategy",
        type=str,
        default="only_stu",
        choices=["only_stu", "only_tch", "intersection", "union"],
        help="top-k候选集合策略",
    )
    parser.add_argument(
        "--weight_mode",
        type=str,
        default="student_p",
        choices=["student_p", "teacher_p", "none"],
        help="候选token的概率权重",
    )
    parser.add_argument("--student_hidden_size", default=768, type=int)
    parser.add_argument("--student_num_layers", default=8, type=int)
    parser.add_argument("--student_use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--teacher_hidden_size", default=768, type=int)
    parser.add_argument("--teacher_num_layers", default=8, type=int)
    parser.add_argument("--teacher_use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否恢复OPD断点")
    parser.add_argument("--thinking_ratio", type=float, default=0.0, help="开启thinking的轨迹比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", help="使用SwanLab记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-OPD")
    parser.add_argument("--wandb_run_name", type=str, default="", help="自定义SwanLab运行名称")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="cloud",
        choices=["cloud", "local", "offline"],
        help="SwanLab记录模式；local可用swanlab watch本地查看",
    )
    parser.add_argument("--wandb_logdir", type=str, default="swanlog", help="SwanLab本地日志目录")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--debug_interval", type=int, default=20)
    args = parser.parse_args()
    validate_args(args)

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    else:
        args.device = resolve_device(args.device)
    rank = dist.get_rank() if dist.is_initialized() else 0
    setup_seed(args.seed + rank)
    resolve_path(args.save_dir).mkdir(parents=True, exist_ok=True)
    resolve_path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    student_config = MiniMindConfig(
        hidden_size=args.student_hidden_size,
        num_hidden_layers=args.student_num_layers,
        use_moe=bool(args.student_use_moe),
        max_position_embeddings=max(32768, args.max_total_len, args.reference_max_len),
    )
    teacher_config = MiniMindConfig(
        hidden_size=args.teacher_hidden_size,
        num_hidden_layers=args.teacher_num_layers,
        use_moe=bool(args.teacher_use_moe),
        max_position_embeddings=max(32768, args.max_total_len),
    )
    if student_config.vocab_size != teacher_config.vocab_size:
        raise ValueError("OPD requires student and teacher to have the same vocabulary size")
    if args.distill_top_k > student_config.vocab_size:
        Logger(
            f"[OPD WARNING] distill_top_k={args.distill_top_k} exceeds vocab={student_config.vocab_size}; "
            "it will be clamped"
        )

    tokenizer = load_and_validate_tokenizer(
        args.student_tokenizer, args.teacher_tokenizer, student_config.vocab_size
    )
    device_type = args.device.split(":", 1)[0]
    mixed_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if device_type == "mps" and args.dtype == "bfloat16":
        Logger("[OPD WARNING] MPS baseline uses float16 teacher weights; overriding --dtype=bfloat16")
        mixed_dtype = torch.float16
    teacher_dtype = mixed_dtype if device_type in {"cuda", "mps"} else torch.float32
    if device_type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(dtype=mixed_dtype)
    else:
        # MPS autocast support differs across PyTorch/macOS versions. Keep the
        # trainable student in FP32 and only store the frozen teacher in FP16.
        autocast_ctx = nullcontext()

    model = load_native_model(student_config, args.student_model, args.device, "student")
    teacher_model = load_native_model(
        teacher_config, args.teacher_model, args.device, "teacher", teacher_dtype=teacher_dtype
    )
    teacher_model.eval().requires_grad_(False)
    reference_model = None
    if args.lambda_ref > 0:
        reference_model = load_native_model(
            student_config,
            args.reference_model,
            args.device,
            "Base reference",
            teacher_dtype=teacher_dtype,
        )
        reference_model.eval().requires_grad_(False)
    Logger(f"Student params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")
    Logger(f"Teacher params: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f}M")
    if reference_model is not None:
        Logger(
            f"Base reference params: {sum(p.numel() for p in reference_model.parameters()) / 1e6:.3f}M, "
            f"lambda_ref={args.lambda_ref}, replay_ratio={args.reference_replay_ratio}, "
            f"active-step weight={args.lambda_ref / args.reference_replay_ratio:.4f}"
        )

    checkpoint = (
        lm_checkpoint(
            student_config,
            weight=args.save_weight,
            save_dir=str(resolve_path(args.checkpoint_dir)),
        )
        if args.from_resume == 1
        else None
    )
    train_ds = AgentRLDataset(str(resolve_path(args.data_path)), tokenizer, max_length=args.max_total_len)
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if dist.is_initialized() else None
    reference_loader = None
    if reference_model is not None:
        reference_ds = ReferenceReplayDataset(
            str(resolve_path(args.reference_data_path)),
            tokenizer,
            max_length=args.reference_max_len,
            max_samples=args.reference_max_samples,
            sample_stride=args.reference_sample_stride,
        )
        reference_sampler = (
            DistributedSampler(reference_ds, shuffle=False) if dist.is_initialized() else None
        )
        reference_loader = DataLoader(
            reference_ds,
            batch_size=args.reference_batch_size,
            sampler=reference_sampler,
            shuffle=False,
            # Keep one tokenizer/dataset copy on Apple Silicon; replay is only
            # consumed on a fraction of steps, so worker fan-out is not useful.
            num_workers=0,
            pin_memory=(device_type == "cuda"),
            collate_fn=collate_reference_batch,
        )
        Logger(
            f"Base reference replay pool: {len(reference_ds)} conversations, "
            f"{len(reference_loader)} batches"
        )

    def collate_fn(batch):
        return {
            "messages": [item["messages"] for item in batch],
            "tools": [item["tools"] for item in batch],
            "gt": [item["gt"] for item in batch],
        }

    count_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn
    )
    iters = len(count_loader)
    scheduled_batches = min(iters, args.max_train_steps) if args.max_train_steps > 0 else iters
    optimizer_steps_per_epoch = math.ceil(scheduled_batches / args.accumulation_steps)
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and args.dtype == "float16"))

    start_epoch, start_step = 0, 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"]
        start_step = checkpoint.get("step", 0)
        Logger(f"Resuming OPD from epoch={start_epoch}, step={start_step}")

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine = create_rollout_engine(
        engine_type="torch",
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
    )

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = checkpoint.get("wandb_id") if checkpoint else None
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or (
                f"Agent-OPD-K{args.distill_top_k}-{args.top_k_strategy}-"
                f"BS{args.batch_size}x{args.accumulation_steps}-LR{args.learning_rate}"
            ),
            config=vars(args),
            mode=args.wandb_mode,
            logdir=args.wandb_logdir,
            id=wandb_id if args.wandb_mode == "cloud" else None,
            resume="must" if wandb_id and args.wandb_mode == "cloud" else None,
        )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch + rank)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if epoch == start_epoch and start_step > 0 else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=(device_type == "cuda"),
            collate_fn=collate_fn,
        )
        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: skipping {skip} completed batches")
        opd_train_epoch(
            epoch,
            loader,
            len(loader) + skip,
            rollout_engine,
            teacher_model,
            reference_model,
            reference_loader,
            student_config,
            start_step=skip,
            wandb=wandb,
        )
        start_step = 0

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
