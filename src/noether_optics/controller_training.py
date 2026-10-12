"""Fixed-budget learned fitting for the study, without persistence or stage access.

    complete training cohort -> independent normalized episode gradients
        -> equal mean at one parameter state -> one Adam update -> retained history

The caller must load the pinned base, supply its matching calibration/atmospheres,
and persist this record through the study ledger. No validation/evaluation data
are generated here. Returned numerical failures never expose a usable model.
"""
from __future__ import annotations

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .controller_physics import _identity, episode_config, paired_open_loop, training_cases
from .controller_protocol import study_plan
from .simulation import initialize_weights, loop


def _episode_kernel(cfg: dict, cal: dict):
    """Differentiate only post-burn variance; retain a full-output finite check."""
    config = deepcopy(cfg)
    predictor = jnp.zeros((config["modes"], 2))

    def objective(params, phases, seed, open_variance):
        outputs = loop(config, cal, phases, seed, "learned", predictor,
                       fault="none", fallback=False, params=params,
                       differentiable_noise=True)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in outputs.values()]))
        variance = jnp.mean(outputs["variance"][config["burn_in"]:])
        return variance / open_variance, finite

    return jax.jit(jax.value_and_grad(objective, has_aux=True))


def _arrays(tree):
    return tuple(np.array(value, copy=True) for value in tree)


def _finite(tree) -> bool:
    return all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(tree))


def fit_learned_controller(base: dict, study: dict, candidate: dict,
                           cal: dict, episodes: list[dict]) -> dict:
    """Fit one declared learned candidate, with no early stopping or restarts.

    Each episode is exactly {case, phases}. Identities must cover the complete
    training cohort; arrays must have its full frame count. Clean open-loop
    denominators are calculated here from the same phases, never supplied by a
    caller. All inputs are checked before initialization or gradient execution.

    The returned record contains initial/updated parameter checkpoints and every
    executed episode's loss and gradient, including partial nonfinite failures.
    Only status='ready' has non-null parameters. Unexpected exceptions propagate;
    the stage adapter must preserve interrupted evidence and manage resumption.
    This function acquires no timings and writes no files.
    """
    cases = training_cases(study)
    if (_identity(candidate) not in {_identity(row) for row in study_plan(study)["candidates"]}
            or candidate["controller"] != "learned"):
        raise ValueError("Expected a declared learned candidate")
    if base["sensor"] != "pyramid":
        raise ValueError("Learned study fitting requires the pyramid sensor")
    shape = (study["training"]["frames"], base["n"], base["n"])
    expected = {_identity(case) for case in cases}
    found = {}
    for episode in episodes:
        if set(episode) != {"case", "phases"}:
            raise ValueError("Learned episodes accept only case and phase inputs")
        key = _identity(episode["case"])
        if key not in expected or key in found:
            raise ValueError("Foreign or duplicate learned training episode")
        phases = np.asarray(episode["phases"])
        if (phases.shape != shape or not np.issubdtype(phases.dtype, np.floating)
                or not np.isfinite(phases).all()):
            raise ValueError("Incomplete or nonfinite real training phases")
        found[key] = np.array(phases, copy=True)
    if len(found) != len(cases):
        raise ValueError("Incomplete learned training cohort")

    configs, denominators = [], []
    for case in cases:
        cfg = episode_config(base, study, "training", case["photons"], case["wind_scale"], candidate)
        opened = paired_open_loop(found[_identity(case)], cfg, cal, "none")
        raw = np.asarray(opened["variance"])
        denominator = float(raw[cfg["burn_in"]:].mean())
        if not np.isfinite(raw).all() or not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Training open-loop variance must be finite and positive")
        configs.append(cfg)
        denominators.append(denominator)

    params = initialize_weights(study["seeds"]["initialization"])
    result = {"status": "failed", "error": None, "candidate": deepcopy(candidate),
              "cases": cases, "open_variances": np.asarray(denominators),
              "checkpoints": [_arrays(params)], "steps": [],
              "final": None, "parameters": None}

    def fail(message):
        result["error"] = message
        return result

    if not _finite(params):
        return fail("Nonfinite initialization")
    optimizer = optax.adam(study["training"]["learned_learning_rate"])
    state = optimizer.init(params)
    # Kernels are reused across seeds in each regime, with one reverse-mode tape
    # alive at a time. The complete cohort is never traced as one giant objective.
    kernels = {}
    for case, cfg in zip(cases, configs):
        key = (case["photons"], case["wind_scale"])
        if key not in kernels:
            kernels[key] = _episode_kernel(cfg, cal)

    def evaluate(parameters):
        record = {"episode_losses": [], "episode_outputs_finite": [], "episode_gradients": []}
        for case, denominator in zip(cases, denominators):
            key = (case["photons"], case["wind_scale"])
            (loss, finite), gradient = jax.block_until_ready(kernels[key](
                parameters, jnp.asarray(found[_identity(case)]), case["seed"], denominator))
            record["episode_losses"].append(float(loss))
            record["episode_outputs_finite"].append(bool(finite))
            record["episode_gradients"].append(_arrays(gradient))
            if not bool(finite) or not _finite((loss, gradient)):
                return record, False
        return record, True

    for step in range(study["training"]["learned_steps"]):
        record, finite = evaluate(params)
        record["step"] = step
        result["steps"].append(record)
        if not finite:
            return fail(f"Nonfinite training episode at step {step}")
        # Every derivative above was taken at the SAME parameters. Taking one
        # optimizer update per episode would silently implement another protocol.
        gradient = tuple(jnp.asarray(np.mean(np.stack(parts), axis=0))
                         for parts in zip(*record["episode_gradients"]))
        loss = float(np.mean(record["episode_losses"]))
        norm = float(optax.tree.norm(gradient))
        record.update(mean_loss=loss, mean_gradient=_arrays(gradient), gradient_norm=norm)
        if not _finite((loss, gradient, norm)):
            return fail(f"Nonfinite mean loss or gradient at step {step}")
        updates, state = optimizer.update(gradient, state, params)
        params = optax.apply_updates(params, updates)
        result["checkpoints"].append(_arrays(params))
        if not _finite((state, params)):
            # Retain the failed parameter/state values without making the previous
            # checkpoint an eligible candidate. The caller must bind a failure.
            record["failed_optimizer_state"] = jax.tree.map(lambda x: np.array(x, copy=True), state)
            return fail(f"Nonfinite optimizer state or parameters at step {step}")

    # Score the actual final parameters, not the pre-update loss of the last step.
    final, finite = evaluate(params)
    result["final"] = final
    if not finite:
        return fail("Nonfinite final training objective or gradient")
    final["mean_loss"] = float(np.mean(final["episode_losses"]))
    if not np.isfinite(final["mean_loss"]):
        return fail("Nonfinite final mean loss")
    result.update(status="ready", parameters=_arrays(params))
    return result
