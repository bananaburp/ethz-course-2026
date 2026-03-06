from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

from exercises.ex3 import *

class SO100TrackEnv(gym.Env):
    xml_path: Path
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, xml_path: Path, render_mode=None):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Bonus: moving target velocity (world frame) — must be initialized before _get_obs()
        self.target_vel = np.zeros(3)
        self._target_bounds_low = np.array([0.2, -0.2, 0.1])
        self._target_bounds_high = np.array([0.4, 0.2, 0.4])

        # Define Observation and Action Spaces
        obs = self._get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None

        # Timestep & Episode
        self.sim_timestep = self.model.opt.timestep # 0.002s (500 Hz)
        self.ctrl_decimation = 50 # makes control frequency 10 Hz
        self.ctrl_timestep = self.sim_timestep * self.ctrl_decimation # 0.1
        self.max_episode_length_s = 10
        self.max_episode_length = int(self.max_episode_length_s / self.ctrl_timestep) # 100 steps per episode
        self.current_step = 0

        # Deafult robot home position
        self.default_qpos = np.array([0.0, -1.57, 1.0, 1.0, 0.0, 0.02239])

        # Evaluation metrics
        self.ee_tracking_error = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        mujoco.mj_resetData(self.model, self.data)

        # Reset Robot to home position
        self.data.qpos[:] = reset_robot(self.default_qpos)
        mujoco.mj_forward(self.model, self.data)

        # Reset target position around robot base
        base_pos = self.data.body("Base").xpos.copy()
        self.data.mocap_pos[0] = reset_target_position(base_pos)

        # Bonus: sample a small random target velocity (m/s in world frame)
        self.target_vel = np.random.uniform(-0.05, 0.05, size=(3,))

        self.current_step = 0
        return self._get_obs(), {}

    def _process_action(self, action):
        return process_action(action, self.model.jnt_range)

    def compute_reward(self):
        return compute_reward(self.ee_tracking_error)

    def _update_target(self):
        """Bonus: move target with constant velocity, reflecting at workspace boundaries."""
        base_pos = self.data.body("Base").xpos.copy()
        new_pos = self.data.mocap_pos[0] + self.target_vel * self.ctrl_timestep
        rel = new_pos - base_pos
        for i in range(3):
            if rel[i] < self._target_bounds_low[i] or rel[i] > self._target_bounds_high[i]:
                self.target_vel[i] *= -1
            rel[i] = np.clip(rel[i], self._target_bounds_low[i], self._target_bounds_high[i])
        self.data.mocap_pos[0] = base_pos + rel

    def step(self, action):
        self.data.ctrl[:] = self._process_action(action)
        for _ in range(self.ctrl_decimation):
            mujoco.mj_step(self.model, self.data)
        self._update_target()
        self.ee_tracking_error = np.linalg.norm(self.data.site("ee_site").xpos - self.data.mocap_pos[0])
        reward = self.compute_reward()

        terminated = False
        truncated = False
        self.current_step += 1
        if self.current_step >= self.max_episode_length:
            truncated = True
        obs = self._get_obs()
        
        if self.render_mode == "human":
            self.render()

        # Extra info as metrics for evaluation
        info = {"ee_tracking_error": self.ee_tracking_error.item()}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        qpos = self.data.qpos.flat[:].copy()
        ee_pos_w = self.data.site("ee_site").xpos.copy()
        ee_rot_w = self.data.site("ee_site").xmat.reshape(3, 3)
        base_pos_w = self.data.body("Base").xpos.copy()
        base_rot_w = self.data.body("Base").xmat.reshape(3, 3)
        target_pos_w = self.data.mocap_pos[0].copy()        
        return get_obs(qpos, ee_pos_w, ee_rot_w, base_pos_w, base_rot_w, target_pos_w, self.target_vel)

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Update the viewer's copy of the data
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None