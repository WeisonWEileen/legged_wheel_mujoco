import os
import numpy as np
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium import spaces


DEFAULT_CAMERA_CONFIG = {
    "distance": 3.0,
}


class MujocoGripEnv(MujocoRobotEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }

    def __init__(self,
                 xml_file=os.path.abspath(os.path.join(os.getcwd(), "./asset/main_scene.xml")),
                 n_substeps=50,
                 **kwargs):

        self.init_joint_qpos = {
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
        }

        kwargs['initial_qpos'] = self.init_joint_qpos
        kwargs['model_path'] = xml_file
        kwargs['n_substeps'] = n_substeps

        self.initial_gripper_ctrl = 0.0

        super().__init__(n_actions=8, **kwargs)

        self._set_action_space()

        EzPickle.__init__(
            self,
            xml_file,
            n_substeps,
            **kwargs,
        )


    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        pos_ctrl = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        # print("pos ctrl: ", pos_ctrl)

        self._mujoco_step(pos_ctrl)
        self._step_callback()

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()

        info = {}

        terminated = False

        truncated = False

        reward = 0

        return obs, reward, terminated, truncated, info



    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space



    def _get_obs(self):
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "pinch").copy()

        robot_qpos = np.squeeze(np.array([self._utils.get_joint_qpos(self.model, self.data, name) for name in self._model_names.joint_names[0:7]]))
        # print(self._model_names.joint_names[0:7])

        gripper_state = np.array([self.data.ctrl[-1]])

        obs = np.concatenate(
            [
                robot_qpos,
                gripper_state
            ]
        )

        achieved_goal = grip_pos

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }


    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)

        self.data.ctrl[-1] = self.initial_gripper_ctrl

        self._mujoco.mj_forward(self.model, self.data)

        self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)


    def _sample_goal(self):
        self.goal = np.random.randn(3)
        return self.goal.copy()