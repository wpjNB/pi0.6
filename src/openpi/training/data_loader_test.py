import dataclasses

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_recap_fields_dataset_merges_sidecar_fields(tmp_path):
    model_config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(model_config, 3)
    fields_path = tmp_path / "lerobot_fields.npz"
    np.savez(
        fields_path,
        advantage_indicator=np.array([True, False, True]),
        use_advantage=np.array([True, True, False]),
        is_human_intervention=np.array([False, True, False]),
    )

    merged = _data_loader.ReCAPFieldsDataset(dataset, fields_path)

    assert merged[0]["advantage_indicator"].item() is True
    assert merged[1]["use_advantage"].item() is True
    assert merged[1]["is_human_intervention"].item() is True
    assert merged[2]["use_advantage"].item() is False


def test_recap_fields_dataset_rejects_length_mismatch(tmp_path):
    model_config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(model_config, 3)
    fields_path = tmp_path / "lerobot_fields.npz"
    np.savez(
        fields_path,
        advantage_indicator=np.array([True, False]),
        use_advantage=np.array([True, True]),
        is_human_intervention=np.array([False, True]),
    )

    try:
        _data_loader.ReCAPFieldsDataset(dataset, fields_path)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected length mismatch to raise ValueError.")


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


# ---------------------------------------------------------------------------
# π0.6-MEM: wall-clock history sampling
# ---------------------------------------------------------------------------


class _StubLeRobotDataset:
    """Minimal LeRobot-like dataset with an explicit fps and feature map.

    Two episodes of ``ep_len`` frames each. ``timestamp`` is a uniform grid at
    ``fps`` so history targets can be checked analytically.
    """

    def __init__(self, num_episodes: int = 2, ep_len: int = 200, fps: float = 20.0, num_cameras: int = 1):
        self.fps = fps
        self._ep_len = ep_len
        self._num_cameras = num_cameras
        self.episode_data_index = {"from": np.arange(num_episodes) * ep_len}
        self.meta = type(
            "Meta",
            (),
            {
                "fps": fps,
                "features": {
                    f"observation.images.cam_{c}": {"dtype": "video"} for c in range(num_cameras)
                }
                | {"observation.state": {"dtype": "float64"}},
            },
        )()

    def __len__(self):
        return self._ep_len * len(self.episode_data_index["from"])

    def __getitem__(self, index):
        episode_index = index // self._ep_len
        frame_index = index % self._ep_len
        sample = {
            f"observation.images.cam_{c}": np.full((4, 4, 3), index, dtype=np.uint8)
            for c in range(self._num_cameras)
        }
        sample.update(
            {
                "observation.state": np.full(3, float(index), dtype=np.float32),
                "timestamp": np.float32(frame_index / self.fps),
                "frame_index": np.int64(frame_index),
                "episode_index": np.int64(episode_index),
            }
        )
        return sample


def test_mem_stride_derived_from_wall_clock():
    """The frame stride must follow fps so the horizon is fps-invariant."""
    ds20 = _data_loader.MEMLeRobotDataset(_StubLeRobotDataset(fps=20.0), num_frames=6, frame_stride_sec=1.0)
    ds50 = _data_loader.MEMLeRobotDataset(_StubLeRobotDataset(fps=50.0), num_frames=6, frame_stride_sec=1.0)

    assert ds20.fps == 20.0
    assert ds20.stride == 20  # 1 s x 20 fps
    assert ds50.stride == 50  # 1 s x 50 fps

    # Both cover the same wall-clock window: (K-1) * stride_sec = 5 s.
    assert (ds20.K - 1) * ds20.stride / ds20.fps == 5.0
    assert (ds50.K - 1) * ds50.stride / ds50.fps == 5.0


def test_mem_history_offsets_match_paper_grid():
    """K-1 history frames land exactly on the 1 s grid (oldest -> newest)."""
    ds = _data_loader.MEMLeRobotDataset(_StubLeRobotDataset(fps=20.0), num_frames=6, frame_stride_sec=1.0)

    indices, valid = ds._history_indices(episode_id=0, current_frame=100)

    offsets = [int(i) - 100 for i in indices]
    assert offsets == [-100, -80, -60, -40, -20]  # 5 s..1 s ago at 20 fps
    assert valid.all()


def test_mem_episode_start_is_padded_and_masked():
    """History before the episode start is clamped and flagged invalid."""
    ds = _data_loader.MEMLeRobotDataset(_StubLeRobotDataset(fps=20.0), num_frames=6, frame_stride_sec=1.0)

    indices, valid = ds._history_indices(episode_id=0, current_frame=30)

    assert indices == [0, 0, 0, 0, 10]
    assert list(valid) == [False, False, False, False, True]


def test_mem_no_cross_episode_leakage():
    """History for the second episode must never reach into the first."""
    ep_len = 200
    ds = _data_loader.MEMLeRobotDataset(
        _StubLeRobotDataset(num_episodes=2, ep_len=ep_len, fps=20.0), num_frames=6, frame_stride_sec=1.0
    )

    indices, valid = ds._history_indices(episode_id=1, current_frame=5)

    assert all(i >= ep_len for i in indices), "history leaked into the previous episode"
    assert indices == [ep_len] * 5
    assert not valid.any()


def test_mem_getitem_returns_history_and_masks():
    ds = _data_loader.MEMLeRobotDataset(
        _StubLeRobotDataset(ep_len=200, fps=20.0, num_cameras=2), num_frames=6, frame_stride_sec=1.0
    )

    sample = ds[100]

    # Camera keys are discovered from the dataset features.
    assert set(sample["image_history"]) == {"cam_0", "cam_1"}
    for key, hist in sample["image_history"].items():
        assert hist.shape == (5, 4, 4, 3), key
    assert set(sample["image_history_masks"]) == {"cam_0", "cam_1"}
    assert sample["image_history_masks"]["cam_0"].shape == (5,)
    assert sample["state_history"].shape == (5, 3)
    assert sample["frame_index"] == 100  # unchanged pass-through


def test_mem_single_frame_is_noop():
    """K=1 must behave exactly like the unwrapped dataset."""
    ds = _data_loader.MEMLeRobotDataset(_StubLeRobotDataset(), num_frames=1)
    sample = ds[7]
    assert "image_history" not in sample
    assert "state_history" not in sample


def test_mem_rejects_invalid_arguments():
    stub = _StubLeRobotDataset()
    for kwargs in ({"num_frames": 0}, {"frame_stride_sec": 0.0}, {"frame_stride_sec": -1.0}):
        try:
            _data_loader.MEMLeRobotDataset(stub, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")

    try:
        _data_loader.MEMLeRobotDataset(stub, time_source="bogus")
    except ValueError as exc:
        assert "time_source" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an unknown time_source")
