"""Shared visual encoder, behavior cloning, and conditional DDPM over action chunks."""

import math

import torch
from torch import nn


class ObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(),
        )

    def forward(self, image, state):
        # Known-color markers make target and end-effector positions observable.
        # Extract their centroids from pixels; no simulator target coordinate is
        # passed to the policy. The CNN still receives the full RGB image.
        red, green, blue = image[:, 0], image[:, 1], image[:, 2]
        goal = (green > red * 1.5) & (green > blue * 1.5) & (green > 0.5)
        tip = (red > green * 1.5) & (red > blue * 1.5) & (red > 0.5)
        axis = torch.linspace(-1, 1, image.shape[-1], device=image.device)
        x_grid = axis.view(1, 1, -1)
        y_grid = axis.view(1, -1, 1)

        def centroid(mask):
            weights = mask.float()
            total = weights.sum(dim=(1, 2)).clamp_min(1)
            return torch.stack(
                [(weights * x_grid).sum(dim=(1, 2)) / total,
                 (weights * y_grid).sum(dim=(1, 2)) / total], dim=1
            )

        markers = torch.cat([centroid(goal), centroid(tip)], dim=1)
        return torch.cat([self.visual(image), state, markers], dim=-1)


class BehaviorCloning(nn.Module):
    def __init__(self, prediction_horizon=4, observation_horizon=2):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.observation_horizon = observation_horizon
        self.encoder = ObservationEncoder()
        self.head = nn.Sequential(
            nn.Linear(72 * observation_horizon, 128), nn.ReLU(),
            nn.Linear(128, prediction_horizon * 2)
        )

    def forward(self, image, state):
        batch, horizon = image.shape[:2]
        features = self.encoder(image.flatten(0, 1), state.flatten(0, 1))
        return self.head(features.reshape(batch, horizon * 72)).reshape(-1, self.prediction_horizon, 2)


class DiffusionPolicy(nn.Module):
    def __init__(self, prediction_horizon=4, observation_horizon=2, diffusion_steps=8):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.observation_horizon = observation_horizon
        self.diffusion_steps = diffusion_steps
        self.encoder = ObservationEncoder()
        self.denoiser = nn.Sequential(
            nn.Linear(72 * observation_horizon + prediction_horizon * 2 + 16, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, prediction_horizon * 2),
        )
        # Cosine schedule reaches almost pure noise at the final step, so
        # ancestral sampling can correctly begin from N(0, I).
        grid = torch.arange(diffusion_steps + 1, dtype=torch.float32) / diffusion_steps
        bars = torch.cos((grid + 0.008) / 1.008 * math.pi / 2).square()
        bars = bars / bars[0]
        betas = (1 - bars[1:] / bars[:-1]).clamp(0.0001, 0.999)
        alphas = 1 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def predict_noise(self, image, state, noisy_actions, timestep):
        # Sinusoidal time embedding keeps inference defined at every diffusion step.
        half = 8
        frequency = torch.exp(
            torch.arange(half, device=image.device, dtype=torch.float32) * (-4.0 / half)
        )
        phase = timestep.float().unsqueeze(1) * frequency.unsqueeze(0)
        time_embedding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        batch, horizon = image.shape[:2]
        features = self.encoder(image.flatten(0, 1), state.flatten(0, 1)).reshape(batch, horizon * 72)
        inputs = torch.cat([features, noisy_actions.flatten(1), time_embedding], dim=1)
        return self.denoiser(inputs).reshape(-1, self.prediction_horizon, 2)

    def loss(self, image, state, actions, mask):
        batch = actions.shape[0]
        timestep = torch.randint(self.diffusion_steps, (batch,), device=actions.device)
        noise = torch.randn_like(actions)
        bar = self.alpha_bars[timestep].view(-1, 1, 1)
        noisy = bar.sqrt() * actions + (1 - bar).sqrt() * noise
        prediction = self.predict_noise(image, state, noisy, timestep)
        errors = (prediction - noise).square() * mask.unsqueeze(-1)
        return errors.sum() / (mask.sum().clamp_min(1) * actions.shape[-1])

    @torch.no_grad()
    def sample(self, image, state, generator=None):
        shape = (image.shape[0], self.prediction_horizon, 2)
        x = torch.randn(shape, device=image.device, generator=generator)
        for step in range(self.diffusion_steps - 1, -1, -1):
            t = torch.full((shape[0],), step, device=image.device, dtype=torch.long)
            predicted_noise = self.predict_noise(image, state, x, t)
            bar = self.alpha_bars[step]
            clean = ((x - (1 - bar).sqrt() * predicted_noise) / bar.sqrt()).clamp(-1, 1)
            if step:
                previous_bar = self.alpha_bars[step - 1]
                # Deterministic DDIM is less jittery for receding-horizon control.
                x = previous_bar.sqrt() * clean + (1 - previous_bar).sqrt() * predicted_noise
            else:
                x = clean
        return x

