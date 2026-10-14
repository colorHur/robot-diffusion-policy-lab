"""Two-link planar reaching task with image and joint-state observations."""

import math

import numpy as np
from PIL import Image, ImageDraw


class ReachEnv:
    image_size = 32
    action_limit = 0.15
    horizon = 35
    success_radius = 0.10

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.angles = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)
        self.steps = 0

    @staticmethod
    def points(angles):
        a, b = float(angles[0]), float(angles[1])
        elbow = np.array([math.cos(a), math.sin(a)], dtype=np.float32)
        tip = elbow + 0.7 * np.array([math.cos(a + b), math.sin(a + b)], dtype=np.float32)
        return elbow, tip

    def reset(self):
        self.angles = self.rng.uniform(-math.pi, math.pi, size=2).astype(np.float32)
        radius = self.rng.uniform(0.35, 1.55)
        direction = self.rng.uniform(-math.pi, math.pi)
        self.target = (radius * np.array([math.cos(direction), math.sin(direction)])).astype(
            np.float32
        )
        self.steps = 0
        return self.observe()

    def distance(self):
        return float(np.linalg.norm(self.points(self.angles)[1] - self.target))

    def observe(self):
        image = Image.new("RGB", (self.image_size, self.image_size), (12, 16, 24))
        draw = ImageDraw.Draw(image)

        def xy(point):
            return (round(16 + float(point[0]) * 8), round(16 - float(point[1]) * 8))

        elbow, tip = self.points(self.angles)
        center, joint, end, goal = map(xy, ([0, 0], elbow, tip, self.target))
        draw.line([center, joint, end], fill=(220, 220, 220), width=2)
        for point, color, radius in (
            (center, (140, 140, 140), 1),
            (joint, (140, 140, 140), 1),
            (end, (250, 90, 90), 2),
            (goal, (80, 240, 120), 2),
        ):
            x, y = point
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        proprio = np.array(
            [math.sin(self.angles[0]), math.cos(self.angles[0]),
             math.sin(self.angles[1]), math.cos(self.angles[1])], dtype=np.float32
        )
        return np.asarray(image, dtype=np.uint8), proprio

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -self.action_limit, self.action_limit)
        self.angles = ((self.angles + action + math.pi) % (2 * math.pi) - math.pi).astype(
            np.float32
        )
        self.steps += 1
        success = self.distance() < self.success_radius
        done = success or self.steps >= self.horizon
        return self.observe(), success, done

    def expert_action(self):
        """Damped least-squares inverse Jacobian controller; used only for demonstrations."""
        a, b = map(float, self.angles)
        _, tip = self.points(self.angles)
        error = self.target - tip
        jacobian = np.array(
            [[-math.sin(a) - 0.7 * math.sin(a + b), -0.7 * math.sin(a + b)],
             [math.cos(a) + 0.7 * math.cos(a + b), 0.7 * math.cos(a + b)]],
            dtype=np.float32,
        )
        damping = 0.06
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping * np.eye(2, dtype=np.float32), error
        )
        return np.clip(delta, -self.action_limit, self.action_limit).astype(np.float32)

