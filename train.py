#pip install tensorflow==1.15
#Install stable-baselines as described in the documentation

import model
from model import CNN1
from utils import get_exp_name, max_exp_idx, load_model, make_vec_envs
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3 import PPO, DQN
from bcq import BCQ
from policies import BCQCnnPolicy
import numpy as np
import os

n_steps = 0
log_dir = './'
best_mean_reward, n_steps = -np.inf, 0

# def callback(_locals, _globals):
#     """
#     Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
#     :param _locals: (dict)
#     :param _globals: (dict)
#     """
#     global n_steps, best_mean_reward
#     # Print stats every 1000 calls
#     if (n_steps + 1) % 10 == 0:
#         x, y = ts2xy(load_results(log_dir), 'timesteps')
#         if len(x) > 100:
#            #pdb.set_trace()
#             mean_reward = np.mean(y[-100:])
#             print(x[-1], 'timesteps')
#             print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

#             # New best model, we save the agent here
#             if mean_reward > best_mean_reward:
#                 best_mean_reward = mean_reward
#                 # Example for saving best model
#                 print("Saving new best model")
#                 _locals['self'].save(os.path.join(log_dir, 'best_model.pkl'))
#             else:
#                 print("Saving latest model")
#                 _locals['self'].save(os.path.join(log_dir, 'latest_model.pkl'))
#         else:
#             print('{} monitor entries'.format(len(x)))
#             pass
#     n_steps += 1
#     # Returning False will stop training early
#     return True


def main(game, representation, model_type, experiment, steps, n_cpu, render, logging, **kwargs):
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    resume = kwargs.get('resume', False)
    if 'buffer_size' in kwargs:
        buffer_size = kwargs.get('buffer_size', 1e6)
        
    if representation == 'wide':
        # policy = FullyConvPolicyBigMap
        pass
        if game == "sokoban":
            # policy = FullyConvPolicySmallMap
            pass
    else:
        # policy = CustomPolicyBigMap
        
        if game == "sokoban":
            if model_type == 'BCQ':
                policy_kwargs = dict(
                    features_extractor_class=CNN1,
                )
                policy = BCQCnnPolicy
    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10
    n = max_exp_idx(exp_name)
    global log_dir
    if not resume:
        n = n + 1
    log_dir = 'runs/{}_{}_{}'.format(exp_name, n, 'log')
    if not resume:
        os.mkdir(log_dir)
    else:
        model = load_model(log_dir)
    kwargs = {
        **kwargs,
        'render_rank': 0,
        'render': render,
    }
    used_dir = log_dir
    if not logging:
        used_dir = None
    env = make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs)
    if not resume or model is None:
        if model_type == 'BCQ':
            model = BCQ(policy, env, buffer_size=buffer_size, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./runs")
    else:
        model.set_env(env)
    if not logging:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name, log_interval=400)
    else:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name, log_interval=400)

################################## MAIN ########################################
game = 'sokoban'
representation = 'narrow'
experiment = None
steps = 1e7
render = False
logging = True
n_cpu = 4
model_type = 'BCQ'
kwargs = {
    'resume': False,
    'change_percentage' : 0.3,
    'buffer_size' : int(1e6)
}

if __name__ == '__main__':
    main(game, representation, model_type, experiment, steps, n_cpu, render, logging, **kwargs)
