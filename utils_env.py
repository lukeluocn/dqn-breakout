from typing import (
    List,
    Optional,
    Tuple,
)

import base64
from collections import deque
import pathlib

from IPython import display as ipydisplay
import numpy as np
from PIL import Image
import torch

from vendor.atari_wrappers import make_atari, wrap_deepmind
from utils_types import (
    GymImg,
    GymObs,
    TensorObs,
    TensorStack4,
    TensorStack5,
    TorchDevice,
)
from utils_drl import Agent

# HTML_TEMPLATE is a template element for displaying an mp4 video
HTML_TEMPLATE = """<video alt="{alt}" autoplay loop controls style="height: 400px;">
  <source src="data:video/mp4;base64,{data}" type="video/mp4" />
</video>"""


class MyEnv(object):

    def __init__(self, device: TorchDevice, seed: Optional[int] = None) -> None:
        """A seed can be provided to make the environment deterministic."""
        env_raw = make_atari("BreakoutNoFrameskip-v4")
        self.__env = wrap_deepmind(env_raw, episode_life=True)
        self.__device = device
        if seed is not None:
            self.__env.seed(seed)

    def reset(
            self,
            render: bool = False,
    ) -> Tuple[List[TensorObs], float, List[GymImg]]:
        """reset resets and initializes the underlying gym environment."""
        self.__env.reset()

        init_reward = 0.
        observations = []
        frames = []
        for _ in range(5): # no-op
            obs, reward, done = self.step(0)
            observations.append(obs)
            init_reward += reward
            if done:
                return self.reset(render)
            if render:
                frames.append(self.get_frame())

        return observations, init_reward, frames

    def step(self, action: int) -> Tuple[TensorObs, int, bool]:
        """step forwards an action to the environment and returns the newest
        observation, the reward, and an bool value indicating whether the
        episode is terminated."""
        action = action + 1 if not action == 0 else 0
        obs, reward, done, _ = self.__env.step(action)
        return self.to_tensor(obs), reward, done

    def get_frame(self) -> GymImg:
        """get_frame renders the current game frame."""
        return Image.fromarray(self.__env.render(mode="rgb_array"))

    @staticmethod
    def to_tensor(obs: GymObs) -> TensorObs:
        """to_tensor converts an observation to a torch tensor."""
        return torch.from_numpy(obs).view(1, 84, 84)

    @staticmethod
    def get_action_dim() -> int:
        """get_action_dim returns the reduced number of actions."""
        return 3

    @staticmethod
    def get_action_meanings() -> List[str]:
        """get_action_meanings returns the actual meanings of the reduced
        actions."""
        return ["NOOP", "RIGHT", "LEFT"]

    @staticmethod
    def get_eval_lives() -> int:
        """get_eval_lives returns the number of lives to consume in an
        evaluation round."""
        return 5

    @staticmethod
    def make_state(obs_queue: deque) -> TensorStack4:
        """make_state makes up a state given an obs queue."""
        return torch.cat(list(obs_queue)[1:]).unsqueeze(0)

    @staticmethod
    def make_folded_state(obs_queue: deque) -> TensorStack5:
        """make_folded_state makes up an n_state given an obs queue."""
        return torch.cat(list(obs_queue)).unsqueeze(0)

    @staticmethod
    def show_video(path_to_mp4: str) -> None:
        """show_video creates an HTML element to display the given mp4 video in
        IPython."""
        mp4 = pathlib.Path(path_to_mp4)
        video_b64 = base64.b64encode(mp4.read_bytes())
        html = HTML_TEMPLATE.format(alt=mp4, data=video_b64.decode("ascii"))
        ipydisplay.display(ipydisplay.HTML(data=html))

    def evaluate(
            self,
            obs_queue: deque,
            agent: Agent,
            num_episode: int = 3,
    ) -> Tuple[
        float,
        List[GymImg],
    ]:
        """evaluate uses the given agent to run the game for a few episodes and
        returns the average reward and the captured frames."""
        ep_rewards = []
        frames = []
        for _ in range(self.get_eval_lives() * num_episode):
            observations, ep_reward, _frames = self.reset(render=True)
            for obs in observations:
                obs_queue.append(obs)
            frames.extend(_frames)
            done = False

            while not done:
                state = self.make_state(obs_queue).to(self.__device).float()
                action = agent.run(state)
                obs, reward, done = self.step(action)

                ep_reward += reward
                obs_queue.append(obs)
                frames.append(self.get_frame())

            ep_rewards.append(ep_reward)

        return np.sum(ep_rewards) / num_episode, frames
