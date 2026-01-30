#!/usr/bin/env python3
import multiprocessing as mp
import signal
from dataclasses import dataclass
from queue import Empty, Full
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
import gym_hil  # noqa: F401
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.constants import ACTION


# -----------------------------
# Feature normalization
# -----------------------------

@dataclass(frozen=True)
class SimpleSpec:
    shape: Tuple[int, ...]
    dtype: str | None = None


def is_dtype_literal(s: str) -> bool:
    return s in {
        "float32", "float64", "float16",
        "int8", "int16", "int32", "int64",
        "uint8", "bool",
    }


def to_spec(x):
    if hasattr(x, "shape"):
        return x
    if isinstance(x, dict) and "shape" in x:
        shp = tuple(int(v) for v in x["shape"])
        dt = x.get("dtype")
        return SimpleSpec(shape=shp, dtype=dt if isinstance(dt, str) else None)
    return x


def unwrap_feature_to_leaf(x, all_features: dict):
    x = to_spec(x)
    if hasattr(x, "shape"):
        return x

    if isinstance(x, str):
        if is_dtype_literal(x):
            raise TypeError(f"dtype literal '{x}' appeared without shape dict.")
        if x not in all_features:
            raise KeyError(f"Feature reference '{x}' not found in meta.features")
        return unwrap_feature_to_leaf(all_features[x], all_features)

    if isinstance(x, dict):
        for preferred in ("continuous", "cont", "value", "vector", "action"):
            if preferred in x:
                return unwrap_feature_to_leaf(x[preferred], all_features)
        return unwrap_feature_to_leaf(next(iter(x.values())), all_features)

    raise TypeError(f"Unsupported feature type: {type(x)}")


def normalize_feature_tree(x, all_features: dict):
    x = to_spec(x)
    if hasattr(x, "shape"):
        return x
    if isinstance(x, str):
        if is_dtype_literal(x):
            raise TypeError(f"dtype literal '{x}' appeared where feature spec expected.")
        return normalize_feature_tree(all_features[x], all_features)
    if isinstance(x, dict):
        return {k: normalize_feature_tree(v, all_features) for k, v in x.items()}
    raise TypeError(f"Unsupported feature tree type: {type(x)}")


def infer_dataset_obs_keys_from_features(features: dict) -> Tuple[str | None, list[str]]:
    keys = list(features.keys())
    state_key = "observation.state" if "observation.state" in keys else None
    image_keys = [k for k in keys if k.startswith("observation.images.")]
    if not image_keys:
        image_keys = [k for k in keys if k.startswith("observation.image.")]
    return state_key, image_keys


# -----------------------------
# Obs conversion
# -----------------------------

def to_chw_float01(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img)
    if t.ndim == 3 and t.shape[-1] in (1, 3, 4):
        t = t.permute(2, 0, 1)
    t = t.float()
    if t.max() > 1.0:
        t = t / 255.0
    return t


def obs_to_state_dict(
    obs: Dict[str, Any],
    state_key: str | None,
    image_keys: list[str],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if state_key is not None:
        out[state_key] = torch.from_numpy(obs["agent_pos"]).float()

    if image_keys:
        cam_dict: Dict[str, np.ndarray] = obs["pixels"]
        cam_names = sorted(cam_dict.keys())
        if len(cam_names) < len(image_keys):
            cam_names = cam_names + [cam_names[-1]] * (len(image_keys) - len(cam_names))
        for i, k in enumerate(image_keys):
            out[k] = to_chw_float01(cam_dict[cam_names[i]])
    return out


def batchify_state(state: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.unsqueeze(0).to(device) for k, v in state.items()}


# -----------------------------
# Action mapping: dataset 4D -> Panda 7D
# dataset action treated as [x,y,z,grasp]
# -----------------------------

def ensure_4d_action(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if a.shape[0] != 4:
        if a.shape[0] > 4:
            a = a[:4]
        else:
            a = np.concatenate([a, np.zeros((4 - a.shape[0],), dtype=np.float32)], axis=0)
    return a


def expand_4d_to_panda_7d(a4: np.ndarray) -> np.ndarray:
    a4 = ensure_4d_action(a4)
    x, y, z, g = a4
    return np.array([x, y, z, 0.0, 0.0, 0.0, g], dtype=np.float32)


# -----------------------------
# Learner subprocess
# -----------------------------

def run_learner(
    transitions_queue: mp.Queue,
    parameters_queue: mp.Queue,
    shutdown_event: mp.Event,
    offline_repo_id: str,
    learner_device: str,
    batch_size: int,
    lr: float,
    online_capacity: int,
    log_every: int = 25,
    send_every: int = 25,
):
    offline_dataset = LeRobotDataset(repo_id=offline_repo_id)
    all_features = offline_dataset.meta.features

    input_features_raw = {k: v for k, v in all_features.items() if k.startswith("observation.")}
    input_features = {k: normalize_feature_tree(v, all_features) for k, v in input_features_raw.items()}
    action_feat = unwrap_feature_to_leaf(all_features[ACTION], all_features)

    policy_cfg = SACConfig(
        device=learner_device,
        input_features=input_features,
        output_features={ACTION: action_feat},
    )
    policy = SACPolicy(policy_cfg)
    policy.train()
    policy.to(torch.device(learner_device))

    state_keys = list(input_features.keys())
    online_buffer = ReplayBuffer(capacity=online_capacity, device=learner_device, state_keys=state_keys)
    offline_buffer = ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=offline_dataset,
        device=learner_device,
        state_keys=state_keys,
    )

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    step = 0
    print(f"[LEARNER] ready | offline={len(offline_buffer)} online_cap={online_capacity}")

    while not shutdown_event.is_set():
        try:
            transitions = transitions_queue.get(timeout=0.1)
            for tr in transitions:
                # IMPORTANT: your ReplayBuffer requires 'truncated'
                online_buffer.add(**tr)
        except Empty:
            pass

        if len(online_buffer) < policy.config.online_step_before_learning:
            continue

        half = batch_size // 2
        use_offline = len(offline_buffer) >= half

        online_bs = batch_size if not use_offline else half
        online_batch = online_buffer.sample(online_bs)

        if use_offline:
            offline_batch = offline_buffer.sample(half)
            batch = {}
            for k in online_batch.keys():
                if k in offline_batch:
                    batch[k] = torch.cat([online_batch[k], offline_batch[k]], dim=0)
                else:
                    batch[k] = online_batch[k]
        else:
            batch = online_batch

        loss = policy.forward(batch, model="critic")["loss_critic"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        policy.update_target_networks()
        policy.update_temperature()

        step += 1
        if step % log_every == 0:
            print(
                f"[LEARNER] step={step} loss={loss.item():.4f} "
                f"online={len(online_buffer)} offline={len(offline_buffer)}"
            )

        if step % send_every == 0:
            try:
                sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
                parameters_queue.put_nowait(sd)
            except Full:
                pass

    print("[LEARNER] finished")


# -----------------------------
# Main: actor with rendering
# -----------------------------

def main():
    mp.set_start_method("spawn", force=True)

    offline_repo_id = "aractingi/franka_sim_pick_lift_6"
    learner_device = "mps"   # or "cuda"
    actor_device = "cpu"

    env_id = "gym_hil/PandaPickCubeBase-v0"

    batch_size = 256
    lr = 3e-4
    online_capacity = 100_000

    max_episodes = 10
    max_steps_per_episode = 300

    transitions_queue = mp.Queue(maxsize=10)
    parameters_queue = mp.Queue(maxsize=2)
    shutdown_event = mp.Event()

    def signal_handler(sig, frame=None):
        print(f"\n[MAIN] signal {sig} -> shutdown")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    learner_p = mp.Process(
        target=run_learner,
        args=(
            transitions_queue,
            parameters_queue,
            shutdown_event,
            offline_repo_id,
            learner_device,
            batch_size,
            lr,
            online_capacity,
        ),
    )
    learner_p.start()

    # Actor policy in main
    ds = LeRobotDataset(repo_id=offline_repo_id)
    feats = ds.meta.features
    state_key, image_keys = infer_dataset_obs_keys_from_features(feats)

    input_features_raw = {k: v for k, v in feats.items() if k.startswith("observation.")}
    input_features = {k: normalize_feature_tree(v, feats) for k, v in input_features_raw.items()}
    action_feat = unwrap_feature_to_leaf(feats[ACTION], feats)
    dataset_action_dim = int(action_feat.shape[0])
    print(f"[ACTOR] env={env_id} | dataset_action_dim={dataset_action_dim}")

    policy_cfg = SACConfig(
        device=actor_device,
        input_features=input_features,
        output_features={ACTION: action_feat},
    )
    policy_actor = SACPolicy(policy_cfg)
    policy_actor.eval()
    policy_actor.to(torch.device(actor_device))

    env = gym.make(env_id, image_obs=True)
    print(f"[ACTOR] action_space={env.action_space}")
    print(f"[ACTOR] using dataset image keys: {image_keys}")

    # Matplotlib render setup
    obs, info = env.reset(seed=0)
    cam = sorted(obs["pixels"].keys())[0]
    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(obs["pixels"][cam])
    ax.set_title(f"Gym-HIL live | {env_id} | cam={cam}")
    ax.axis("off")

    torch_device_actor = torch.device(actor_device)

    try:
        for ep in range(max_episodes):
            if shutdown_event.is_set():
                break

            obs, info = env.reset(seed=ep)
            state = obs_to_state_dict(obs, state_key, image_keys)

            episode_transitions = []
            ep_return = 0.0

            for t in range(max_steps_per_episode):
                # pull weights if available
                try:
                    sd = parameters_queue.get_nowait()
                    policy_actor.load_state_dict(sd, strict=True)
                except Empty:
                    pass

                with torch.no_grad():
                    a4 = policy_actor.select_action(batchify_state(state, torch_device_actor)).squeeze(0).cpu().numpy()
                a4 = ensure_4d_action(a4)

                env_action = expand_4d_to_panda_7d(a4)
                next_obs, reward, terminated, truncated, info = env.step(env_action)

                done = bool(terminated)  # keep separated; buffer needs both
                trunc = bool(truncated)

                ep_return += float(reward)

                next_state = obs_to_state_dict(next_obs, state_key, image_keys)

                # IMPORTANT: include truncated
                episode_transitions.append(
                    {
                        "state": {k: v.clone() for k, v in state.items()},
                        "action": torch.from_numpy(a4.astype(np.float32)).float(),
                        "reward": torch.tensor(float(reward), dtype=torch.float32),
                        "next_state": {k: v.clone() for k, v in next_state.items()},
                        "done": torch.tensor(float(done), dtype=torch.float32),
                        "truncated": torch.tensor(float(trunc), dtype=torch.float32),
                        "complementary_info": {"is_intervention": False},
                    }
                )

                # render
                im.set_data(next_obs["pixels"][cam])
                fig.canvas.draw()
                fig.canvas.flush_events()

                state = next_state
                if done or trunc:
                    break

            try:
                transitions_queue.put_nowait(episode_transitions)
            except Full:
                pass

            print(f"[ACTOR] episode={ep+1}/{max_episodes} steps={len(episode_transitions)} return={ep_return:.3f}")

    finally:
        shutdown_event.set()
        env.close()
        plt.ioff()
        plt.show()

        learner_p.join(timeout=10)
        if learner_p.is_alive():
            learner_p.terminate()

        print("[MAIN] done")


if __name__ == "__main__":
    main()
