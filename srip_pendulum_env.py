# Replicated from original gymnasium code with modifications for SRIP pendulum task.
# Copyright 2013 OpenAI, Inc.
# Original code:
# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

DEFAULT_THETA_LOW = -np.pi
DEFAULT_THETA_HIGH = np.pi
DEFAULT_THETADOT_LOW = -1.0
DEFAULT_THETADOT_HIGH = 1.0
DEFAULT_G = 10.0
DEFAULT_M = 1.0
DEFAULT_L = 1.0
DEFAULT_B = 1.0
DEFAULT_DT = 0.1
DEFAULT_N_SUBSTEPS = 20         # 200Hz integration / 10Hz policy = 20 sub-steps
DEFAULT_DT = 0.1                # policy dt stays 0.1s
DEFAULT_SETPOINT_PERIOD = 5.0
DEFAULT_Q1 = 1.0
DEFAULT_Q2 = 0.1
DEFAULT_R = 0.001
DEFAULT_MAX_TIME = 20
DEFAULT_MAX_TORQUE = 20.0

class PendulumEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    DEFAULT_THETA_LOW = -np.pi
    DEFAULT_THETA_HIGH = np.pi
    
    def __init__(self, render_mode: str | None = None, g=DEFAULT_G):
        self.max_speed = 8.0
        self.max_torque = DEFAULT_MAX_TORQUE
        self.dt = DEFAULT_DT
        self.g = g
        self.m = DEFAULT_M
        self.l = DEFAULT_L
        self.b = DEFAULT_B

        self.render_mode = render_mode
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, action):
        u = action
        th, thdot = self.state
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l**2) * u) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, False, False, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([self.DEFAULT_THETA_HIGH, DEFAULT_THETADOT_HIGH], dtype=np.float32)
        else:
            theta_low = options.get("theta_low", self.DEFAULT_THETA_LOW)
            theta_high = options.get("theta_high", self.DEFAULT_THETA_HIGH)
            thetadot_low = options.get("thetadot_low", DEFAULT_THETADOT_LOW)
            thetadot_high = options.get("thetadot_high", DEFAULT_THETADOT_HIGH)
            theta_low = utils.verify_number_and_cast(theta_low)
            theta_high = utils.verify_number_and_cast(theta_high)
            thetadot_low = utils.verify_number_and_cast(thetadot_low)
            thetadot_high = utils.verify_number_and_cast(thetadot_high)
            high = np.array([theta_high, thetadot_high], dtype=np.float32)
            low = np.array([theta_low, thetadot_low], dtype=np.float32)
            self.state = self.np_random.uniform(low=low, high=high)
            self.last_u = None
            if self.render_mode == "human":
                self.render()
            return self._get_obs(), {}

        low = np.array([self.DEFAULT_THETA_LOW, DEFAULT_THETADOT_LOW], dtype=np.float32)
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
            else:
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class SRIPPendulumEnv(PendulumEnv):
    DEFAULT_THETA_LOW = -np.pi
    DEFAULT_THETA_HIGH = np.pi
    
    def __init__(
        self,
        render_mode=None,
        g=DEFAULT_G,
        m=DEFAULT_M,
        l=DEFAULT_L,
        b=DEFAULT_B,
        q1=DEFAULT_Q1,
        q2=DEFAULT_Q2,
        r=DEFAULT_R,
        dt=DEFAULT_DT,
        n_substeps=DEFAULT_N_SUBSTEPS,
        setpoint_period=DEFAULT_SETPOINT_PERIOD,
        nonlinear=True,
    ):
        super().__init__(render_mode=render_mode, g=g)
        self.max_torque = DEFAULT_MAX_TORQUE
        self.dt = dt
        self.n_substeps = n_substeps
        self.substep_dt = dt / n_substeps
        self.m = m
        self.l = l
        self.b = b
        self.q1 = q1
        self.q2 = q2
        self.r = r
        self.nonlinear = nonlinear
        self.setpoint_period = setpoint_period
        self.setpoint_steps = max(1, int(np.round(self.setpoint_period / self.dt)))
        self.step_counter = 0
        self.target_angle = 0.0
        self.randomize_setpoint = True

        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
            shape=(1,),
            dtype=np.float32,
        )
        # Encode target as (cos, sin) to avoid discontinuity at ±π
        # obs: [cos(θ), sin(θ), θ̇, cos(θ₀), sin(θ₀)]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -self.max_speed, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0,  self.max_speed,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _derivatives(self, th, thdot, u):
        if self.nonlinear:
            ddth = (
                (self.g / self.l) * np.sin(th)
                - (self.b / (self.m * self.l**2)) * thdot
                + u / (self.m * self.l**2)
            )
        else:
            ddth = (
                (self.g / self.l) * th
                - (self.b / (self.m * self.l**2)) * thdot
                + u / (self.m * self.l**2)
            )
        return ddth

    def step(self, action):
        u = np.clip(action, -self.max_torque, self.max_torque)[0]
        self.last_u = u

        th, thdot = self.state

        # Sub-stepped Euler integration
        for _ in range(self.n_substeps):
            ddth = self._derivatives(th, thdot, u)
            thdot = np.clip(thdot + ddth * self.substep_dt, -self.max_speed, self.max_speed)
            th = th + thdot * self.substep_dt

        self.state = np.array([th, thdot], dtype=np.float32)

        error = angle_normalize(th - self.target_angle)
        costs = self.q1 * (error**2) + self.q2 * (thdot**2) + self.r * (u**2)

        self.step_counter += 1
        if self.randomize_setpoint and self.step_counter % self.setpoint_steps == 0:
            self.target_angle = self._sample_setpoint()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), -costs, False, False, {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array(
            [
                np.cos(theta),
                np.sin(theta),
                thetadot,
                np.cos(self.target_angle),   # encode as unit circle — no discontinuity
                np.sin(self.target_angle),
            ],
            dtype=np.float32,
        )
        
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = {} if options is None else options

        self.m = float(options.get("m", self.m))
        self.l = float(options.get("l", self.l))
        self.g = float(options.get("g", self.g))
        self.b = float(options.get("b", self.b))
        self.q1 = float(options.get("q1", self.q1))
        self.q2 = float(options.get("q2", self.q2))
        self.r = float(options.get("r", self.r))
        self.dt = float(options.get("dt", self.dt))
        self.setpoint_period = float(options.get("setpoint_period", self.setpoint_period))
        self.setpoint_steps = max(1, int(np.round(self.setpoint_period / self.dt)))

        self.randomize_setpoint = bool(options.get("randomize_setpoint", True))
        requested_setpoint = options.get("setpoint")
        if requested_setpoint is None:
            self.target_angle = self._sample_setpoint()
        else:
            self.target_angle = float(requested_setpoint)

        theta_low = float(options.get("theta_low", self.DEFAULT_THETA_LOW))
        theta_high = float(options.get("theta_high", self.DEFAULT_THETA_HIGH))
        thetadot_low = float(options.get("thetadot_low", DEFAULT_THETADOT_LOW))
        thetadot_high = float(options.get("thetadot_high", DEFAULT_THETADOT_HIGH))

        self.state = self.np_random.uniform(
            low=np.array([theta_low, thetadot_low], dtype=np.float32),
            high=np.array([theta_high, thetadot_high], dtype=np.float32),
        )
        self.last_u = None
        self.step_counter = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}
    
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
            else:
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = self.l * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))

        target_end = (rod_length, 0)
        target_end = pygame.math.Vector2(target_end).rotate_rad(self.target_angle + np.pi / 2)
        target_end = (int(target_end[0] + offset), int(target_end[1] + offset))
        gfxdraw.aacircle(self.surf, target_end[0], target_end[1], int(rod_width / 2), (255, 0, 0))
        gfxdraw.filled_circle(self.surf, target_end[0], target_end[1], int(rod_width / 2), (255, 0, 0))

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def _sample_setpoint(self):
        return self.np_random.uniform(low=self.DEFAULT_THETA_LOW, high=self.DEFAULT_THETA_HIGH)


class LinearSRIPPendulumEnv(SRIPPendulumEnv):
    DEFAULT_THETA_LOW = -np.pi / 4
    DEFAULT_THETA_HIGH = np.pi / 4
    
    def __init__(
        self,
        render_mode: str | None = None,
        g=DEFAULT_G,
        m=DEFAULT_M,
        l=DEFAULT_L,
        b=DEFAULT_B,
        q1=DEFAULT_Q1,
        q2=DEFAULT_Q2,
        r=DEFAULT_R,
        dt=DEFAULT_DT,
        setpoint_period=DEFAULT_SETPOINT_PERIOD,
    ):
        super().__init__(
            render_mode=render_mode,
            g=g,
            m=m,
            l=l,
            b=b,
            q1=q1,
            q2=q2,
            r=r,
            dt=dt,
            setpoint_period=setpoint_period,
            nonlinear=False,
        )
        # Provide θ directly — linear dynamics are NOT periodic,
        # so cos/sin encoding creates ambiguity
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -self.max_speed, -np.pi], dtype=np.float32),
            high=np.array([ np.pi,  self.max_speed,  np.pi], dtype=np.float32),
            dtype=np.float32,
        )
        
    def _get_obs(self):
        theta, thetadot = self.state
        # Wrap θ so the observation stays bounded
        theta_wrapped = angle_normalize(theta)
        return np.array(
            [theta_wrapped, thetadot, self.target_angle],
            dtype=np.float32,
        )


class NonlinearSRIPPendulumEnv(SRIPPendulumEnv):
    DEFAULT_THETA_LOW = -np.pi 
    DEFAULT_THETA_HIGH = np.pi 
    def __init__(
        self,
        render_mode: str | None = None,
        g=DEFAULT_G,
        m=DEFAULT_M,
        l=DEFAULT_L,
        b=DEFAULT_B,
        q1=DEFAULT_Q1,
        q2=DEFAULT_Q2,
        r=DEFAULT_R,
        dt=DEFAULT_DT,
        setpoint_period=DEFAULT_SETPOINT_PERIOD,
    ):
        super().__init__(
            render_mode=render_mode,
            g=g,
            m=m,
            l=l,
            b=b,
            q1=q1,
            q2=q2,
            r=r,
            dt=dt,
            setpoint_period=setpoint_period,
            nonlinear=True,
        )


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi