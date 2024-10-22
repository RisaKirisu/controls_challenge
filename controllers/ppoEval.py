from . import BaseController
import numpy as np
import torch
from PPO_control.ActorCritic import ActorCritic
import warnings
warnings.filterwarnings("ignore")

class Controller(BaseController):
  def __init__(self) -> None:
    super().__init__()
    self.device = torch.device('cpu')
    policy_model = ActorCritic(obs_dim=4, obs_seq_len=20, target_dim=31, action_dim=1, has_continuous_action=True, action_scale=2).to(self.device)
    policy_params = torch.load('PPO_control/checkpoints/exp0828-2_step700.pt')['model_state_dict']
    policy_model.load_state_dict(policy_params)
    self.policy_model = policy_model

    self.concat_state_history = []

    # self.p = 0.3
    # self.i = 0.05
    # self.d = -0.1
    # self.error_integral = None
    # self.prev_error = None

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    target_lataccel = np.array([[target_lataccel]])
    current_lataccel = np.array([[current_lataccel]])
    state = np.array([list(state)])
    future_plan = np.array([[list(x) for x in future_plan]]).transpose((0, 2, 1))
    steer = np.zeros_like(target_lataccel)

    if self.concat_state_history:
        self.concat_state_history[-1][:, :, -1] = current_lataccel[:, np.newaxis]
        # if len(self.concat_state_history) < self.policy_model.obs_seq_len:
        #   if self.error_integral is None:
        #     self.error_integral = np.zeros_like(target_lataccel)
        #     self.prev_error = np.zeros_like(target_lataccel)
            
        #   error = (target_lataccel - current_lataccel)
        #   self.error_integral += error
        #   error_diff = error - self.prev_error
        #   self.prev_error = error
        #   steer = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # else:
        past_obs = np.concatenate(self.concat_state_history[-self.policy_model.obs_seq_len:], axis=1)
        past_obs = torch.from_numpy(past_obs).float().to(self.device)

        future_plan = future_plan[:, :9, [1, 3, 0]]
        target = np.zeros((state.shape[0], 31))
        target[:, 0] = current_lataccel
        target[:, 1:3] = state[:, :2]
        target[:, 3] = target_lataccel
        target[:, 4: future_plan.shape[1] * future_plan.shape[2] + 4] = future_plan.reshape((future_plan.shape[0], -1))
        target = torch.from_numpy(target).float().to(self.device)

        action, _, _= self.policy_model.act(past_obs, target, True)
        steer = action

    self.concat_state_history.append(np.column_stack([state[:, :2], steer, np.zeros_like(steer)])[:, np.newaxis])

    return steer.item()
