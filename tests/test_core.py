import numpy as np
import torch

from robot_lab.data import collect, load
from robot_lab.env import ReachEnv
from robot_lab.models import BehaviorCloning, DiffusionPolicy
from robot_lab.run import bc_loss


def test_expert_reaches_some_targets():
    successes = 0
    for seed in range(30):
        env = ReachEnv(seed)
        env.reset()
        for _ in range(env.horizon):
            _, success, done = env.step(env.expert_action())
            if done:
                successes += int(success)
                break
    assert successes >= 20


def test_temporal_dataset_and_models(tmp_path):
    path = tmp_path / "demo.npz"
    summary = collect(path, episodes=3, seed=0, prediction_horizon=4, observation_horizon=2)
    images, states, actions, masks = load(path)
    assert summary["samples"] == len(images) == len(states) == len(actions) == len(masks)
    assert images.shape[1:] == (2, 32, 32, 3)
    assert states.shape[1:] == (2, 4)
    assert actions.shape[1:] == (4, 2)
    assert masks.shape[1:] == (4,)
    assert np.max(np.abs(actions)) <= ReachEnv.action_limit + 1e-6
    with np.load(path) as archive:
        ids = archive["episode_ids"]
    for episode_id in range(3):
        episode = np.flatnonzero(ids == episode_id)
        first, last = episode[0], episode[-1]
        np.testing.assert_array_equal(images[first, 0], images[first, 1])
        np.testing.assert_array_equal(states[first, 0], states[first, 1])
        np.testing.assert_array_equal(masks[last], [True, False, False, False])
        np.testing.assert_array_equal(actions[last, 1:], 0)
        if len(episode) > 1:
            np.testing.assert_array_equal(images[last, 0], images[last - 1, 1])

    image = torch.from_numpy(images[:2].copy()).permute(0, 1, 4, 2, 3).float() / 255
    state = torch.from_numpy(states[:2].copy())
    action = torch.from_numpy(actions[:2].copy()) / ReachEnv.action_limit
    mask = torch.from_numpy(masks[:2].copy()).float()
    bc = BehaviorCloning()
    diffusion = DiffusionPolicy()
    prediction = bc(image, state)
    assert prediction.shape == (2, 4, 2)
    assert torch.isfinite(bc_loss(prediction, action, mask))
    assert torch.isfinite(diffusion.loss(image, state, action, mask))
    assert diffusion.sample(image, state).shape == (2, 4, 2)


def test_masked_loss_ignores_padded_actions():
    prediction = torch.zeros((1, 4, 2))
    target = torch.tensor([[[1., 1.], [100., 100.], [100., 100.], [100., 100.]]])
    mask = torch.tensor([[1., 0., 0., 0.]])
    assert bc_loss(prediction, target, mask).item() == 1.0

