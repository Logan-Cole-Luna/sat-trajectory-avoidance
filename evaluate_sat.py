#!/usr/bin/env python3
"""Deterministic evaluation for the trained PPO satellite avoidance model.

This script mirrors the training configuration, runs multiple seeded trials,
reports aggregate statistics, and emits a JSON summary that downstream tooling
(e.g. `utils/eval/plot_metrics.py`) can consume.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO

# Direct imports without optional guards
import yaml
import pandas as pd
import psutil
from tqdm import tqdm

from utils.eval.satellite_avoidance_env import SatelliteAvoidanceEnv
from utils.eval.eval_plotting import generate_policy_plots


DEFAULT_MODEL_PATH = "models/satellite_avoidance_model.zip"
DEFAULT_CONFIG_PATH = "configs/eval.yaml"
DEFAULT_OUTPUT_DIR = "results/evaluation"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for deterministic evaluation and return the namespace."""
    parser = argparse.ArgumentParser(
        description="Evaluate the trained PPO satellite avoidance policy with deterministic seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the trained PPO model (.zip)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Training/evaluation configuration YAML")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes to run")
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=12345, help="Base random seed for deterministic trials")
    parser.add_argument("--device", default="cpu", help="Computation device for the policy (cpu/cuda)")
    parser.add_argument("--output", default=None, help="Optional JSON output path (defaults to timestamped file)")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic action selection (enabled by default)",
    )
    parser.add_argument(
        "--scenario-save-limit",
        type=int,
        default=10,
        help="Number of scenario blueprints to include for reproducibility",
    )
    # Progress bar control
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable tqdm progress bars during evaluation",
    )
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    if not args.deterministic:
        # We always evaluate deterministically; keep flag for backwards compatibility.
        args.deterministic = True
    return args


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config from path if it exists; return empty dict otherwise."""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def build_reward_params(train_cfg: Dict[str, Any]) -> Dict[str, float]:
    """Extract reward shaping parameters from train config with sensible defaults."""
    reward_cfg = train_cfg.get("reward", {}) if isinstance(train_cfg, dict) else {}
    def _get(name: str, default: float) -> float:
        try:
            return float(reward_cfg.get(name, default))
        except (TypeError, ValueError):
            return float(default)
    return {
        "collision_penalty": _get("collision_penalty", 500.0),
        "survival_reward": _get("survival_reward", 0.02),
        "distance_shaping_coeff": _get("distance_shaping_coeff", 0.0002),
        "coast_bonus": _get("coast_bonus", 0.05),
        "delta_v_linear_cost": _get("delta_v_linear_cost", 6.0),
        "delta_v_quadratic_cost": _get("delta_v_quadratic_cost", 0.5),
        "large_burn_penalty": _get("large_burn_penalty", 3.0),
        "smoothness_penalty": _get("smoothness_penalty", 0.005),
        "cumulative_dv_penalty": _get("cumulative_dv_penalty", 0.2),
        "projected_miss_reward": _get("projected_miss_reward", 0.0002),
        "milestone_reward": _get("milestone_reward", 0.4),
    }


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch (including CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env(
    cfg: Dict[str, Any],
    reward_params: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[SatelliteAvoidanceEnv, Dict[str, Any], Dict[str, Any]]:
    """Construct a template env and return (env, kwargs used, meta like debris seed)."""
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    test_cfg = cfg.get("test", {}) if isinstance(cfg, dict) else {}

    def _float_from(source: Dict[str, Any], key: str, default: Optional[float]) -> Optional[float]:
        if source is None:
            return default
        try:
            value = source.get(key, default)
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    collision_prob = _float_from(test_cfg, "collision_probability", None)
    if collision_prob is None:
        collision_prob = _float_from(train_cfg.get("collision", {}), "probability", 0.7)
    debris_radius = _float_from(test_cfg, "debris_radius_m", None)
    if debris_radius is None:
        debris_radius = _float_from(train_cfg.get("collision", {}), "debris_radius_m", 0.0)
    guarantee_raw = test_cfg.get("guarantee_collision")
    guarantee_collision = bool(guarantee_raw) if guarantee_raw is not None else False
    collision_ttc = _float_from(test_cfg, "ttc_steps", None)
    if collision_ttc is None:
        collision_ttc = _float_from(train_cfg.get("collision", {}), "ttc_steps", None)
    max_thrust_test = _float_from(test_cfg, "max_thrust", None)
    max_thrust = max_thrust_test if max_thrust_test is not None else _float_from(train_cfg, "max_thrust", 0.05)
    collision_distance = _float_from(test_cfg, "collision_distance_m", None)

    # Deterministic debris template used only during the very first reset.
    debris_positions_seed = rng.integers(low=0, high=2**32 - 1)
    debris_positions = [rng.normal(loc=0.0, scale=1.0, size=3) * 10000.0 for _ in range(100)]

    env_kwargs: Dict[str, Any] = {
        "debris_positions": debris_positions,
        "collision_course_probability": collision_prob,
        "guarantee_collision": guarantee_collision,
        "collision_ttc_steps": None if collision_ttc is None else int(collision_ttc),
        "debris_radius_m": debris_radius,
        "max_thrust": max_thrust,
        "enable_quantitative_metrics": True,
    }
    if collision_distance is not None:
        env_kwargs["collision_distance_m"] = collision_distance
    env_kwargs.update(reward_params)

    env = SatelliteAvoidanceEnv(**env_kwargs)
    meta = {"debris_template_seed": int(debris_positions_seed)}
    return env, env_kwargs, meta


def evaluate_policy(
    name: str,
    env_kwargs: Dict[str, Any],
    seeds: List[int],
    action_fn: Callable[[SatelliteAvoidanceEnv, np.ndarray], np.ndarray],
    *,
    max_steps: int,
    scenario_save_limit: int,
    capture_scenarios: bool = False,
    show_progress: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """Evaluate a policy-like action function deterministically across episodes.

    This runs a fresh `SatelliteAvoidanceEnv` per episode with the provided
    `env_kwargs`, sets deterministic seeds, and steps the environment using the
    supplied `action_fn`. It collects per-episode aggregates (reward, steps,
    delta-v, min distance, fuel consumed, collision flags), per-step time-series
    telemetry, termination reasons, and basic performance metrics (wall time,
    memory usage). Optionally, it captures scenario blueprints for a subset of
    episodes to aid reproducibility.

    Args:
        name: Human-readable policy label used in outputs.
        env_kwargs: Keyword arguments to construct `SatelliteAvoidanceEnv`.
            Note: `debris_positions` is deep-copied per episode to avoid
            cross-episode mutation.
        seeds: Sequence of episode seeds; one episode is run per seed.
        action_fn: Callable taking (env, obs) and returning an action ndarray
            compatible with `env.action_space`.
        max_steps: Safety cap on steps per episode.
        scenario_save_limit: Max number of scenario blueprints to store when
            `capture_scenarios` is True.
        capture_scenarios: If True, record initial scenario state for up to
            `scenario_save_limit` episodes.
        show_progress: When True, displays a tqdm progress bar over episodes.

    Returns:
        A 4-tuple:
            - aggregate (Dict[str, Any]): policy-wide aggregates, including:
                collision_rate, success_rate, avg_reward, avg_steps, avg_delta_v,
                avg_min_distance, avg_fuel_consumed, episodes, wall_time_sec,
                total_steps, avg_wall_time_per_step, memory_usage_* fields.
            - per_episode (Dict[str, Any]): lists of metrics per episode:
                episode_rewards, episode_lengths, episode_delta_v, min_distances,
                fuel_consumed, collisions, seeds, termination_reasons (counts),
                termination_reasons_per_episode, time_series (per-step telemetry).
            - scenarios (Optional[List[Dict[str, Any]]]): list of scenario
                blueprints for the first N episodes if `capture_scenarios` else None.
            - performance (Dict[str, Any]): duplicate of timing/memory metrics for
                convenience, matching fields in `aggregate`.

    Notes:
        - A new environment instance is created and closed per episode.
        - Seeding is applied globally and to the environment/action/observation spaces.
        - Memory usage is sampled from the current process via psutil.
        - Per-episode time-series include elapsed_time, min_distance,
          cumulative_delta_v, and delta_v_step.
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_delta_v: List[float] = []
    episode_min_distance: List[float] = []
    episode_fuel_consumed: List[float] = []
    collision_flags: List[bool] = []
    termination_counter: Counter[str] = Counter()
    scenario_snapshots: List[Dict[str, Any]] = [] if capture_scenarios else []
    episode_termination_reasons: List[Optional[str]] = []
    episode_time_series: List[Dict[str, List[float]]] = []
    episode_reward_components: List[Dict[str, float]] = []
    total_steps_across_episodes = 0

    eval_start_time = time.perf_counter()
    process = psutil.Process(os.getpid())
    memory_start = process.memory_info().rss

    iterator = enumerate(seeds)
    if show_progress:
        iterator = tqdm(
            enumerate(seeds),
            total=len(seeds),
            desc=f"Evaluating {name}",
            unit="episode",
            leave=False,
        )

    for idx, episode_seed in iterator:
        set_global_seed(episode_seed)
        # Fresh environment per evaluation to avoid cross-policy contamination
        env_kwargs_local = dict(env_kwargs)
        env_kwargs_local["debris_positions"] = [
            np.array(pos, dtype=float).copy() for pos in env_kwargs["debris_positions"]
        ]
        env = SatelliteAvoidanceEnv(**env_kwargs_local)
        env.action_space.seed(episode_seed)
        env.observation_space.seed(episode_seed)

        obs, _info = env.reset(seed=episode_seed)
        if capture_scenarios and len(scenario_snapshots) < scenario_save_limit:
            scenario_state = env.get_scenario_state()
            scenario_state = {
                "policy": name,
                "episode_index": idx,
                "seed": episode_seed,
                **scenario_state,
            }
            scenario_snapshots.append(_to_serializable(scenario_state))

        episode_reward = 0.0
        steps = 0
        cumulative_delta_v = 0.0
        min_distance = math.inf
        last_fuel = float(getattr(env, "fuel_mass", 0.0))
        initial_fuel = last_fuel
        collided = False
        termination_reason: Optional[str] = None

        per_step_elapsed: List[float] = []
        per_step_min_distance: List[float] = []
        per_step_cumulative_dv: List[float] = []
        per_step_delta_v: List[float] = []

        while steps < max_steps:
            action = action_fn(env, obs)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            episode_reward += float(reward)
            dv_step = float(info.get("last_delta_v", 0.0))
            cumulative_delta_v += dv_step
            cur_min = info.get("min_distance")
            if cur_min is not None:
                min_distance = min(min_distance, float(cur_min))
            collided = collided or bool(info.get("collision_occurred", False))
            last_fuel = float(info.get("fuel_mass", last_fuel))
            if info.get("termination_reason"):
                termination_reason = str(info["termination_reason"])

            elapsed_time = info.get("elapsed_time")
            if elapsed_time is None:
                time_increment = float(getattr(env, "time_increment", 1.0))
                elapsed_time = float(steps * time_increment)
            per_step_elapsed.append(float(elapsed_time))

            step_min_distance = info.get("min_distance")
            per_step_min_distance.append(float(step_min_distance) if step_min_distance is not None else float("nan"))

            per_step_cumulative_dv.append(float(info.get("cumulative_delta_v", cumulative_delta_v)))
            per_step_delta_v.append(dv_step)
            if done or truncated:
                break
        else:

            # Hit max steps without env termination
            termination_reason = termination_reason or "max_steps"

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        episode_delta_v.append(cumulative_delta_v)
        episode_min_distance.append(min_distance if math.isfinite(min_distance) else float("nan"))
        episode_fuel_consumed.append(max(0.0, initial_fuel - last_fuel))
        collision_flags.append(collided)
        termination = termination_reason or ("collision" if collided else "episode_complete")
        termination_counter[termination] += 1
        episode_termination_reasons.append(termination)
        episode_time_series.append(
            {
                "elapsed_time": per_step_elapsed,
                "min_distance": per_step_min_distance,
                "cumulative_delta_v": per_step_cumulative_dv,
                "delta_v_step": per_step_delta_v,
            }
        )
        # Snapshot per-episode reward components from latest info (env accumulates totals)
        rc = info.get("reward_components") if isinstance(info, dict) else None
        if isinstance(rc, dict):
            try:
                episode_reward_components.append({k: float(v) for k, v in rc.items()})
            except Exception:
                episode_reward_components.append({})
        else:
            episode_reward_components.append({})

        env.close()
        total_steps_across_episodes += steps

    aggregate = _compute_aggregate(
        episode_rewards,
        episode_lengths,
        episode_delta_v,
        episode_min_distance,
        episode_fuel_consumed,
        collision_flags,
    )

    eval_wall_time = time.perf_counter() - eval_start_time
    memory_end = process.memory_info().rss
    memory_delta = memory_end - memory_start
    avg_step_time = (
        eval_wall_time / total_steps_across_episodes
        if total_steps_across_episodes > 0
        else float("nan")
    )

    performance = {
        "wall_time_sec": eval_wall_time,
        "total_steps": int(total_steps_across_episodes),
        "avg_wall_time_per_step": avg_step_time,
        "memory_usage_start_bytes": memory_start,
        "memory_usage_end_bytes": memory_end,
        "memory_delta_bytes": memory_delta,
    }

    aggregate.update(
        {
            "wall_time_sec": eval_wall_time,
            "total_steps": int(total_steps_across_episodes),
            "avg_wall_time_per_step": avg_step_time,
            "memory_usage_start_bytes": memory_start,
            "memory_usage_end_bytes": memory_end,
            "memory_delta_bytes": memory_delta,
        }
    )
    per_episode = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_delta_v": episode_delta_v,
        "min_distances": episode_min_distance,
        "fuel_consumed": episode_fuel_consumed,
        "collisions": collision_flags,
        "seeds": list(seeds),
        "termination_reasons": dict(termination_counter),
        "termination_reasons_per_episode": episode_termination_reasons,
        "time_series": episode_time_series,
        "reward_components": episode_reward_components,
    }
    if capture_scenarios:
        per_episode["scenario_blueprints"] = scenario_snapshots
    scenarios_out = scenario_snapshots if capture_scenarios else None
    return aggregate, per_episode, scenarios_out, performance


def _compute_aggregate(
    rewards: List[float],
    lengths: List[int],
    delta_v: List[float],
    min_distances: List[float],
    fuel_consumed: List[float],
    collisions: List[bool],
) -> Dict[str, Any]:
    """Compute policy-level aggregate metrics from per-episode lists."""
    def _safe_mean(values: List[float]) -> float:
        cleaned = [float(v) for v in values if v is not None and not math.isnan(v)]
        if not cleaned:
            return float("nan")
        return float(np.mean(cleaned))

    collision_rate = float(np.mean(collisions)) if collisions else float("nan")
    return {
        "collision_rate": collision_rate,
        "success_rate": float(1.0 - collision_rate) if not math.isnan(collision_rate) else float("nan"),
        "avg_reward": _safe_mean(rewards),
        "avg_steps": _safe_mean([float(v) for v in lengths]),
        "avg_delta_v": _safe_mean(delta_v),
        "avg_min_distance": _safe_mean(min_distances),
        "avg_fuel_consumed": _safe_mean(fuel_consumed),
        "episodes": len(rewards),
    }


def _to_serializable(obj: Any) -> Any:
    """Convert numpy/scalar types to JSON-safe values; map NaN/inf floats to None."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(obj, (np.int32, np.int64, np.int16)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _slugify(value: str) -> str:
    """Return a filesystem-friendly slug for labels like policy names."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _policy_per_episode_dataframe(policy_name: str, per_episode: Dict[str, Any]) -> Optional[Any]:
    """Build a per-episode metrics DataFrame for a policy from the JSON payload."""

    def _coerce_sequence(seq: Any) -> List[Any]:
        if isinstance(seq, np.ndarray):
            return seq.tolist()
        if isinstance(seq, list):
            return seq
        if isinstance(seq, tuple):
            return list(seq)
        if seq is None:
            return []
        try:
            return list(seq)
        except TypeError:
            return []

    rewards = _coerce_sequence(per_episode.get("episode_rewards"))
    lengths = _coerce_sequence(per_episode.get("episode_lengths"))
    delta_v = _coerce_sequence(per_episode.get("episode_delta_v"))
    min_distances = _coerce_sequence(per_episode.get("min_distances"))
    fuel = _coerce_sequence(per_episode.get("fuel_consumed"))
    collisions = _coerce_sequence(per_episode.get("collisions"))
    seeds = _coerce_sequence(per_episode.get("seeds"))
    terminations = _coerce_sequence(per_episode.get("termination_reasons_per_episode"))

    counts = [len(rewards), len(lengths), len(delta_v), len(min_distances), len(fuel), len(collisions), len(seeds), len(terminations)]
    episode_count = max(counts) if counts else 0
    rows: List[Dict[str, Any]] = []

    def _safe_float(value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            f_value = float(value)
        except (TypeError, ValueError):
            return float("nan")
        if math.isnan(f_value) or math.isinf(f_value):
            return float("nan")
        return f_value

    for idx in range(episode_count):
        row: Dict[str, Any] = {
            "policy": policy_name,
            "episode": idx + 1,
            "reward": _safe_float(rewards[idx]) if idx < len(rewards) else float("nan"),
            "length": None,
            "delta_v": _safe_float(delta_v[idx]) if idx < len(delta_v) else float("nan"),
            "min_distance": _safe_float(min_distances[idx]) if idx < len(min_distances) else float("nan"),
            "fuel_consumed": _safe_float(fuel[idx]) if idx < len(fuel) else float("nan"),
            "collision": None,
            "seed": seeds[idx] if idx < len(seeds) else None,
            "termination_reason": terminations[idx] if idx < len(terminations) else None,
        }

        if idx < len(lengths):
            try:
                row["length"] = int(lengths[idx]) if lengths[idx] is not None else None
            except (TypeError, ValueError):
                row["length"] = None

        if idx < len(collisions):
            coll_val = collisions[idx]
            if coll_val is None:
                row["collision"] = None
            else:
                row["collision"] = bool(coll_val)

        rows.append(row)

    df = pd.DataFrame(rows)
    try:
        df["length"] = pd.to_numeric(df["length"], errors="coerce").astype("Int64")
    except Exception:
        pass
    try:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    except Exception:
        pass
    try:
        df["collision"] = df["collision"].astype("boolean")
    except Exception:
        pass
    return df


def _termination_counts_dataframe(policy_name: str, per_episode: Dict[str, Any]) -> Optional[Any]:
    """Return a DataFrame of termination reason counts for a policy (or empty)."""
    counts = per_episode.get("termination_reasons", {}) or {}
    if not counts:
        return pd.DataFrame(columns=["policy", "reason", "count"])
    rows = []
    for reason, count in dict(counts).items():
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            continue
        rows.append({"policy": policy_name, "reason": str(reason), "count": count_int})
    return pd.DataFrame(rows)


def _policy_time_series_dataframe(policy_name: str, per_episode: Dict[str, Any]) -> Optional[Any]:
    """Expand per-episode time-series into a long-form DataFrame per policy."""
    time_series = per_episode.get("time_series", []) or []
    rows: List[Dict[str, Any]] = []

    def _coerce_float_list(values: Any) -> List[float]:
        if isinstance(values, np.ndarray):
            return [float(v) if v is not None else float("nan") for v in values.tolist()]
        if isinstance(values, (list, tuple)):
            result: List[float] = []
            for v in values:
                if v is None:
                    result.append(float("nan"))
                else:
                    try:
                        result.append(float(v))
                    except (TypeError, ValueError):
                        result.append(float("nan"))
            return result
        return []

    for idx, entry in enumerate(time_series, start=1):
        times = _coerce_float_list(entry.get("elapsed_time"))
        min_d = _coerce_float_list(entry.get("min_distance"))
        cumulative_dv = _coerce_float_list(entry.get("cumulative_delta_v"))
        delta_v_step = _coerce_float_list(entry.get("delta_v_step"))
        length = max(len(times), len(min_d), len(cumulative_dv), len(delta_v_step))
        for step in range(length):
            rows.append(
                {
                    "policy": policy_name,
                    "episode": idx,
                    "step_index": step,
                    "elapsed_time": times[step] if step < len(times) else float("nan"),
                    "min_distance": min_d[step] if step < len(min_d) else float("nan"),
                    "cumulative_delta_v": cumulative_dv[step] if step < len(cumulative_dv) else float("nan"),
                    "delta_v_step": delta_v_step[step] if step < len(delta_v_step) else float("nan"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    try:
        df["elapsed_time"] = pd.to_numeric(df["elapsed_time"], errors="coerce")
    except Exception:
        pass
    try:
        df["min_distance"] = pd.to_numeric(df["min_distance"], errors="coerce")
    except Exception:
        pass
    try:
        df["cumulative_delta_v"] = pd.to_numeric(df["cumulative_delta_v"], errors="coerce")
    except Exception:
        pass
    try:
        df["delta_v_step"] = pd.to_numeric(df["delta_v_step"], errors="coerce")
    except Exception:
        pass
    return df


def _compute_time_series_stats(time_df: Any) -> Optional[Any]:
    """Compute mean/std by time buckets for selected metrics across policies."""
    if time_df is None or time_df.empty:
        return None
    df = time_df.copy()
    df = df.dropna(subset=["elapsed_time"])
    if df.empty:
        return None

    try:
        df["time_bucket"] = df["elapsed_time"].round(1)
    except Exception:
        return None

    metrics = [col for col in ["min_distance", "cumulative_delta_v", "delta_v_step"] if col in df.columns]
    if not metrics:
        return None

    agg = df.groupby(["policy", "time_bucket"]).agg({metric: ["mean", "std"] for metric in metrics})
    agg.columns = [f"{metric}_{stat}" for metric in metrics for stat in ["mean", "std"]]
    agg = agg.reset_index()
    for metric in metrics:
        std_col = f"{metric}_std"
        if std_col in agg.columns:
            agg[std_col] = agg[std_col].fillna(0.0)
    return agg




def _save_metrics_and_plots(base_dir: str, json_path: str, payload: Dict[str, Any]) -> None:
    """Write JSON and CSV artifacts then generate plots for all evaluated policies."""
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2)

    if payload.get("scenarios"):
        scenario_path = os.path.join(base_dir, "scenarios.json")
        with open(scenario_path, "w", encoding="utf-8") as handle:
            json.dump(_to_serializable(payload["scenarios"]), handle, indent=2)

    # Always save metrics and plots; pandas is a required dependency now

    per_episode_dir = os.path.join(base_dir, "per_episode")
    os.makedirs(per_episode_dir, exist_ok=True)

    episode_frames: List[Any] = []
    termination_frames: List[Any] = []
    time_series_frames: List[Any] = []
    reward_component_frames: List[Any] = []
    aggregate_rows: List[Dict[str, Any]] = []

    results = payload.get("results", {}) or {}
    for policy_name, result in results.items():
        per_episode = result.get("per_episode", {}) or {}
        aggregate = result.get("aggregate", {}) or {}

        df_episode = _policy_per_episode_dataframe(policy_name, per_episode)
        if df_episode is not None and not df_episode.empty:
            episode_frames.append(df_episode)
            csv_path = os.path.join(per_episode_dir, f"{_slugify(policy_name)}_per_episode.csv")
            df_episode.to_csv(csv_path, index=False)

        df_termination = _termination_counts_dataframe(policy_name, per_episode)
        if df_termination is not None and not df_termination.empty:
            termination_frames.append(df_termination)

        df_time_series = _policy_time_series_dataframe(policy_name, per_episode)
        if df_time_series is not None and not df_time_series.empty:
            time_series_frames.append(df_time_series)
            ts_path = os.path.join(per_episode_dir, f"{_slugify(policy_name)}_time_series.csv")
            df_time_series.to_csv(ts_path, index=False)

        # Reward components per-episode (if available)
        rc_list = per_episode.get("reward_components", []) or []
        if isinstance(rc_list, list) and len(rc_list) > 0:
            # Normalize dicts to same keys
            all_keys = set()
            for d in rc_list:
                if isinstance(d, dict):
                    all_keys.update(d.keys())
            rows_rc: List[Dict[str, Any]] = []
            for idx, d in enumerate(rc_list, start=1):
                row = {"policy": policy_name, "episode": idx}
                if isinstance(d, dict):
                    for k in all_keys:
                        try:
                            row[k] = float(d.get(k, 0.0))
                        except Exception:
                            row[k] = float("nan")
                rows_rc.append(row)
            df_rc = pd.DataFrame(rows_rc)
            if not df_rc.empty:
                reward_component_frames.append(df_rc)
                rc_path = os.path.join(per_episode_dir, f"{_slugify(policy_name)}_reward_components.csv")
                df_rc.to_csv(rc_path, index=False)

        if aggregate:
            row = {"policy": policy_name}
            row.update(aggregate)
            aggregate_rows.append(row)

    combined_df = pd.concat(episode_frames, ignore_index=True) if episode_frames else None
    termination_df = pd.concat(termination_frames, ignore_index=True) if termination_frames else None
    aggregates_df = pd.DataFrame(aggregate_rows) if aggregate_rows else None
    time_series_df = pd.concat(time_series_frames, ignore_index=True) if time_series_frames else None
    time_series_stats = _compute_time_series_stats(time_series_df) if time_series_df is not None else None
    reward_components_df = pd.concat(reward_component_frames, ignore_index=True) if reward_component_frames else None

    if combined_df is not None and not combined_df.empty:
        combined_df.to_csv(os.path.join(base_dir, "all_policies_per_episode.csv"), index=False)

    if termination_df is not None and not termination_df.empty:
        termination_df.to_csv(os.path.join(base_dir, "termination_reasons.csv"), index=False)

    if aggregates_df is not None and not aggregates_df.empty:
        aggregates_df.to_csv(os.path.join(base_dir, "policy_aggregates.csv"), index=False)

    if time_series_df is not None and not time_series_df.empty:
        time_series_df.to_csv(os.path.join(base_dir, "all_policies_time_series.csv"), index=False)

    if time_series_stats is not None and not time_series_stats.empty:
        time_series_stats.to_csv(os.path.join(base_dir, "time_series_stats.csv"), index=False)

    if reward_components_df is not None and not reward_components_df.empty:
        reward_components_df.to_csv(os.path.join(base_dir, "all_policies_reward_components.csv"), index=False)
        # Also emit simple per-policy means for convenience
        # Identify component columns (exclude policy/episode)
        comp_cols = [c for c in reward_components_df.columns if c not in {"policy", "episode"}]
        if comp_cols:
            rc_means = reward_components_df.groupby("policy")[comp_cols].mean(numeric_only=True).reset_index()
            rc_means.to_csv(os.path.join(base_dir, "reward_component_means.csv"), index=False)

    generate_policy_plots(base_dir, combined_df, termination_df, aggregates_df, time_series_df, time_series_stats)

    # Markdown summary for quick review
    if aggregates_df is not None and not aggregates_df.empty:
        summary_md = os.path.join(base_dir, "summary.md")
        try:
            with open(summary_md, "w", encoding="utf-8") as handle:
                handle.write("# Policy Evaluation Summary\n\n")
                handle.write(f"Summary JSON: `{json_path}`\n\n")
                try:
                    table_text = aggregates_df.to_markdown(index=False)
                except Exception:
                    table_text = aggregates_df.to_csv(index=False)
                handle.write(table_text)
                handle.write("\n")
        except Exception:
            pass


def _prepare_output_locations(args: argparse.Namespace) -> Tuple[str, str]:
    """Resolve output base directory and summary JSON path from CLI args."""
    if args.output:
        output_arg = os.path.normpath(args.output)
        if output_arg.lower().endswith(".json"):
            json_path = output_arg
            base_dir = os.path.splitext(output_arg)[0]
        else:
            base_dir = output_arg
            json_path = os.path.join(base_dir, "summary.json")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"ppo_eval_{timestamp}")
        json_path = os.path.join(base_dir, "summary.json")

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    return base_dir, json_path


def main() -> None:
    """Entry point: evaluate PPO and baselines, then save artifacts and summary."""
    print("Starting Evaluation")
    args = parse_args()
    cfg = load_config(args.config)
    reward_params = build_reward_params(cfg.get("train", {}) if isinstance(cfg, dict) else {})
    rng = np.random.default_rng(args.seed)

    set_global_seed(args.seed)
    env, env_kwargs, env_meta = build_env(cfg, reward_params, rng)
    # We'll spawn fresh environments per policy evaluation; close the template instance.
    env.close()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found at '{args.model}'")

    model = PPO.load(args.model, device=args.device)
    model.set_random_seed(args.seed)
    model.policy.set_training_mode(False)

    seeds = [args.seed + i for i in range(args.episodes)]
    scenario_limit = max(0, int(args.scenario_save_limit))

    def ppo_action_fn(env: SatelliteAvoidanceEnv, obs: np.ndarray) -> np.ndarray:
        """SB3 PPO policy action with deterministic predict."""
        action, _ = model.predict(obs, deterministic=True)
        return action

    def baseline_action_fn(env: SatelliteAvoidanceEnv, obs: np.ndarray) -> np.ndarray:
        """Rule-based baseline action from the environment's built-in heuristic."""
        return np.asarray(env._baseline_policy(obs), dtype=env.action_space.dtype)

    def no_action_fn(env: SatelliteAvoidanceEnv, obs: np.ndarray) -> np.ndarray:
        """No-thrust baseline (all zeros)."""
        return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    def legacy_baseline_action_fn(env: SatelliteAvoidanceEnv, obs: np.ndarray) -> np.ndarray:
        """Simpler legacy heuristic: thrust directly away from the closest debris
        when within distance thresholds; otherwise coast. No TCA/velocity shaping.

        Heuristic:
        - Compute distance to nearest debris; if none, do nothing.
        - Define thresholds using env.collision_distance_m if available; otherwise
          fall back to max(10 km, debris_radius_m).
        - If distance < danger_distance: thrust magnitude 1.0 directly away.
        - Else if distance < caution_distance: thrust magnitude 0.3 away.
        - Else: no thrust.
        """
        # Gather satellite and debris states
        sat_pos = np.asarray(getattr(env, "satellite_position", np.zeros(3)), dtype=float)
        if not getattr(env, "debris_positions", None):
            return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

        # Determine base collision threshold
        base_threshold = getattr(env, "collision_distance_m", None)
        if base_threshold is None:
            base_threshold = max(10000.0, float(getattr(env, "debris_radius_m", 0.0)))
        # Operational bands
        caution_distance = max(2.0 * float(base_threshold), 3000.0)
        danger_distance = max(3.0 * float(base_threshold), 6000.0)

        # Find the closest debris
        nearest_vec = None
        nearest_dist = float("inf")
        for d in getattr(env, "debris_positions", []):
            vec = np.asarray(d, dtype=float) - sat_pos
            dist = float(np.linalg.norm(vec))
            if np.isfinite(dist) and dist < nearest_dist:
                nearest_dist = dist
                nearest_vec = vec

        if nearest_vec is None or not np.isfinite(nearest_dist) or nearest_dist <= 1e-6:
            return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

        # Direction directly away from closest debris
        away_dir = -nearest_vec / max(nearest_dist, 1e-8)

        # Choose thrust magnitude based on distance band
        if nearest_dist < danger_distance:
            mag = 1.0
        elif nearest_dist < caution_distance:
            mag = 0.3
        else:
            mag = 0.0

        action = away_dir * mag
        # Clip to action space bounds and cast
        low = getattr(env.action_space, "low", -1.0)
        high = getattr(env.action_space, "high", 1.0)
        action = np.clip(action, low, high)
        return action.astype(env.action_space.dtype)

    def impulsive_maneuver_action_fn(env: SatelliteAvoidanceEnv, obs: np.ndarray) -> np.ndarray:
        """Sustained lateral-accel avoidance (impulsive-style plan, acceleration-limited).

        The environment maps action directly to acceleration (|a| <= max_thrust),
        so a single-step "impulse" cannot create a large Δv. Here we compute a
        constant-acceleration plan towards the lateral perpendicular direction
        and apply it over many steps until the predicted TCA window elapses or
        the projected miss distance exceeds the safety clearance.

        Plan on first call per episode, then return the cached action for a
        fixed number of steps. Falls back to coasting if no maneuver is needed.
        """
        # Reuse plan if one exists for this episode
        plan = getattr(env, "_imp_plan", None)
        if plan is not None:
            if plan.get("steps_remaining", 0) > 0:
                plan["steps_remaining"] -= 1
                return np.asarray(plan["action"], dtype=env.action_space.dtype)
            # Plan complete; stop thrusting
            return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

        try:
            sat_pos = np.asarray(getattr(env, "satellite_position"), dtype=float)
            sat_vel = np.asarray(getattr(env, "satellite_velocity"), dtype=float)
            debris_positions = list(getattr(env, "debris_positions", []))
            debris_velocities = list(getattr(env, "debris_velocities", [np.zeros(3) for _ in debris_positions]))
            if not debris_positions:
                return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

            # Choose target debris
            tgt_idx = getattr(env, "collision_target_index", None)
            if tgt_idx is None or tgt_idx >= len(debris_positions) or tgt_idx < 0:
                dists = [float(np.linalg.norm(np.asarray(d, dtype=float) - sat_pos)) for d in debris_positions]
                tgt_idx = int(np.argmin(dists)) if dists else 0
            tgt_pos = np.asarray(debris_positions[int(tgt_idx)], dtype=float)
            tgt_vel = np.asarray(debris_velocities[int(tgt_idx)], dtype=float)

            r0 = sat_pos - tgt_pos
            v_rel = sat_vel - tgt_vel
            v2 = float(np.dot(v_rel, v_rel))
            dt = float(getattr(env, "time_increment", 1.0))
            if not np.isfinite(v2) or v2 < 1e-12:
                # If relative motion is tiny, just push away moderately for a short window
                away = r0 / max(np.linalg.norm(r0), 1e-9)
                action = (away * 0.5)
                env._imp_plan = {"action": action, "steps_remaining": int(max(1, math.ceil(5.0 / max(dt, 1e-6))))}
                return np.asarray(action, dtype=env.action_space.dtype)

            # Predict TCA and perpendicular miss now
            t_star = -float(np.dot(r0, v_rel)) / v2
            t_star = float(np.clip(t_star, 0.0, 1200.0))
            v_hat = v_rel / max(np.linalg.norm(v_rel), 1e-9)
            r_perp = r0 - np.dot(r0, v_hat) * v_hat
            miss_now = float(np.linalg.norm(r_perp))

            # Safety threshold and buffer
            base_thresh = getattr(env, "collision_distance_m", None)
            if base_thresh is None:
                base_thresh = max(10000.0, float(getattr(env, "debris_radius_m", 0.0)))
            buffer = float(getattr(env, "safe_zone_buffer_m", 3000.0))
            clearance = float(base_thresh) + buffer

            d_req = max(0.0, clearance - miss_now)
            if d_req <= 0.0 or t_star <= 0.0:
                env._imp_plan = {"action": np.zeros(env.action_space.shape), "steps_remaining": 0}
                return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

            # Constant acceleration needed to produce displacement d_req in t_star: a = 2 d / t^2
            a_req = 2.0 * d_req / max(t_star ** 2, 1e-6)
            max_thrust = float(getattr(env, "max_thrust", 0.05))
            mag = float(np.clip(a_req / max_thrust, 0.0, 1.0))

            # Direction: perpendicular to v_rel, aligned with r_perp
            if np.linalg.norm(r_perp) < 1e-9:
                # Pick any unit vector perpendicular to v_rel
                trial = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(trial, v_hat)) > 0.95:
                    trial = np.array([0.0, 1.0, 0.0])
                n = np.cross(v_hat, trial)
                n_norm = np.linalg.norm(n)
                n = n / max(n_norm, 1e-9)
            else:
                n = r_perp / max(np.linalg.norm(r_perp), 1e-9)

            action = n * mag
            # Clip to bounds
            low = getattr(env.action_space, "low", -1.0)
            high = getattr(env.action_space, "high", 1.0)
            action = np.clip(action, low, high)

            # Number of steps to sustain the burn (up to t_star)
            steps_needed = int(max(1, math.ceil(t_star / max(dt, 1e-6))))
            env._imp_plan = {"action": action, "steps_remaining": steps_needed}
            return action.astype(env.action_space.dtype)
        except Exception:
            env._imp_plan = {"action": np.zeros(env.action_space.shape), "steps_remaining": 0}
            return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    ppo_aggregate, ppo_per_episode, scenarios, ppo_performance = evaluate_policy(
        "ppo",
        env_kwargs,
        seeds,
        ppo_action_fn,
        max_steps=args.max_steps,
        scenario_save_limit=scenario_limit,
        capture_scenarios=True,
        show_progress=args.progress,
    )

    impulsive_aggregate, impulsive_per_episode, _, impulsive_performance = evaluate_policy(
        "baseline_impulsive",
        env_kwargs,
        seeds,
        impulsive_maneuver_action_fn,
        max_steps=args.max_steps,
        scenario_save_limit=scenario_limit,
        capture_scenarios=False,
        show_progress=args.progress,
    )

    baseline_aggregate, baseline_per_episode, _, baseline_performance = evaluate_policy(
        "baseline_rule",
        env_kwargs,
        seeds,
        baseline_action_fn,
        max_steps=args.max_steps,
        scenario_save_limit=scenario_limit,
        capture_scenarios=False,
        show_progress=args.progress,
    )

    #legacy_aggregate, legacy_per_episode, _, legacy_performance = evaluate_policy(
    #    "baseline_legacy",
    #    env_kwargs,
    #    seeds,
    #    legacy_baseline_action_fn,
    #    max_steps=args.max_steps,
    #    scenario_save_limit=scenario_limit,
    #    capture_scenarios=False,
    #    show_progress=args.progress,
    #)

    no_action_aggregate, no_action_per_episode, _, no_action_performance = evaluate_policy(
        "no_action",
        env_kwargs,
        seeds,
        no_action_fn,
        max_steps=args.max_steps,
        scenario_save_limit=scenario_limit,
        capture_scenarios=False,
        show_progress=args.progress,
    )

    output_dir, summary_path = _prepare_output_locations(args)

    payload = {
        "meta": {
            "model_path": args.model,
            "config": args.config,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "device": args.device,
            "seed": args.seed,
            "env_kwargs": _to_serializable(env_kwargs),
            "timestamp": datetime.now().isoformat(),
            **env_meta,
        },
        "aggregate": ppo_aggregate,
        "per_episode": ppo_per_episode,
        "performance": {
            "ppo": ppo_performance,
            "baseline_impulsive": impulsive_performance,
            #"baseline_legacy": legacy_performance,
            "baseline_rule": baseline_performance,
            "no_action": no_action_performance,
        },
        "results": {
            "ppo": {
                "aggregate": ppo_aggregate,
                "per_episode": ppo_per_episode,
                "performance": ppo_performance,
            },
            "baseline_impulsive": {
                "aggregate": impulsive_aggregate,
                "per_episode": impulsive_per_episode,
                "performance": impulsive_performance,
            },
            #"baseline_legacy": {
            #    "aggregate": legacy_aggregate,
            #    "per_episode": legacy_per_episode,
            #    "performance": legacy_performance,
            #},
            "baseline_rule": {
                "aggregate": baseline_aggregate,
                "per_episode": baseline_per_episode,
                "performance": baseline_performance,
            },
            "no_action": {
                "aggregate": no_action_aggregate,
                "per_episode": no_action_per_episode,
                "performance": no_action_performance,
            },
        },
    }

    if scenarios:
        payload["scenarios"] = scenarios

    _save_metrics_and_plots(output_dir, summary_path, payload)

    def _print_policy_summary(label: str, aggregate: Dict[str, Any]) -> None:
        print(f"\n-- {label} --")
        print(f"Episodes:          {aggregate['episodes']}")
        print(
            f"Collision rate:    {aggregate['collision_rate']:.2%}"
            if not math.isnan(aggregate['collision_rate'])
            else "Collision rate:    n/a"
        )
        print(
            f"Success rate:      {aggregate['success_rate']:.2%}"
            if not math.isnan(aggregate['success_rate'])
            else "Success rate:      n/a"
        )
        print(
            f"Average reward:    {aggregate['avg_reward']:.3f}"
            if not math.isnan(aggregate['avg_reward'])
            else "Average reward:    n/a"
        )
        print(
            f"Average steps:     {aggregate['avg_steps']:.2f}"
            if not math.isnan(aggregate['avg_steps'])
            else "Average steps:     n/a"
        )
        print(
            f"Average delta-v:   {aggregate['avg_delta_v']:.3f} m/s"
            if not math.isnan(aggregate['avg_delta_v'])
            else "Average delta-v:   n/a"
        )
        print(
            f"Average min dist.: {aggregate['avg_min_distance']:.2f} m"
            if not math.isnan(aggregate['avg_min_distance'])
            else "Average min dist.: n/a"
        )
        print(
            f"Avg fuel consumed: {aggregate['avg_fuel_consumed']:.3f} kg"
            if not math.isnan(aggregate['avg_fuel_consumed'])
            else "Avg fuel consumed: n/a"
        )
        wall_time = aggregate.get("wall_time_sec")
        if wall_time is not None and not (isinstance(wall_time, float) and math.isnan(wall_time)):
            print(f"Total wall time:  {float(wall_time):.2f} s")
        avg_step_time = aggregate.get("avg_wall_time_per_step")
        if avg_step_time is not None and not (isinstance(avg_step_time, float) and math.isnan(avg_step_time)):
            print(f"Avg step wall time: {float(avg_step_time):.6f} s")
        total_steps = aggregate.get("total_steps")
        if isinstance(total_steps, (int, float)) and not math.isnan(float(total_steps)):
            print(f"Total steps:       {int(total_steps)}")
        mem_end = aggregate.get("memory_usage_end_bytes")
        if isinstance(mem_end, (int, float)) and mem_end is not None:
            print(f"Memory (end):      {float(mem_end) / (1024**2):.2f} MiB")
        mem_delta = aggregate.get("memory_delta_bytes")
        if isinstance(mem_delta, (int, float)) and mem_delta is not None:
            print(f"Memory delta:      {float(mem_delta) / (1024**2):.4f} MiB")

    print("\n=== Evaluation summary ===")
    _print_policy_summary("PPO (trained)", ppo_aggregate)
    _print_policy_summary("Baseline (impulsive ΔV)", impulsive_aggregate)
    #_print_policy_summary("Baseline (legacy)", legacy_aggregate)
    _print_policy_summary("Baseline (rule-based)", baseline_aggregate)
    _print_policy_summary("Baseline (no-action)", no_action_aggregate)
    print(f"\nSummary saved to:  {summary_path}")
    print(f"Artifacts directory: {output_dir}")


if __name__ == "__main__":
    main()
