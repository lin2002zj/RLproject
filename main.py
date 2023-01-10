import gym
import numpy as np
import torch
import os.path as osp
import time
import sys
from torch import Tensor
from agent import Agent
from module import mbpo_epoches, test_agent
from logger import EpochLogger

def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False):
    datestamp = datestamp

    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])
    
    if seed is not None:
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), 
                         exp_name=exp_name)
    return logger_kwargs

def get_mc_return(bias_eval_env, agent, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff):
    final_mc_list = np.zeros(0)
    final_mc_entropy_list = np.zeros(0)
    final_obs_list = []
    final_act_list = []
    while final_mc_list.shape[0] < n_mc_eval:
        o = bias_eval_env.reset()
        reward_list, log_prob_a_tilda_list, obs_list, act_list = [], [], [], []
        r, d, ep_ret, ep_len = 0, False, 0, 0
        discounted_return = 0
        discounted_return_with_entropy = 0
        for i_step in range(max_ep_len):  # run an episode
            with torch.no_grad():
                a, log_prob_a_tilda = agent.get_action_and_logprob_for_bias_evaluation(o)
            obs_list.append(o)
            act_list.append(a)
            o, r, d, _ = bias_eval_env.step(a)
            ep_ret += r
            ep_len += 1
            reward_list.append(r)
            log_prob_a_tilda_list.append(log_prob_a_tilda.item())
            if d or (ep_len == max_ep_len):
                break
        discounted_return_list = np.zeros(ep_len)
        discounted_return_with_entropy_list = np.zeros(ep_len)
        for i_step in range(ep_len - 1, -1, -1):
            # backwards compute discounted return and with entropy for all s-a visited
            if i_step == ep_len - 1:
                discounted_return_list[i_step] = reward_list[i_step]
                discounted_return_with_entropy_list[i_step] = reward_list[i_step]
            else:
                discounted_return_list[i_step] = reward_list[i_step] + gamma * discounted_return_list[i_step + 1]
                discounted_return_with_entropy_list[i_step] = reward_list[i_step] + \
                                                              gamma * (discounted_return_with_entropy_list[i_step + 1] - alpha * log_prob_a_tilda_list[i_step + 1])
        # now we take the first few of these.
        final_mc_list = np.concatenate((final_mc_list, discounted_return_list[:n_mc_cutoff]))
        final_mc_entropy_list = np.concatenate(
            (final_mc_entropy_list, discounted_return_with_entropy_list[:n_mc_cutoff]))
        final_obs_list += obs_list[:n_mc_cutoff]
        final_act_list += act_list[:n_mc_cutoff]
    return final_mc_list, final_mc_entropy_list, np.array(final_obs_list), np.array(final_act_list)

def log_bias_evaluation(bias_eval_env, agent, logger, \
                    max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff):
    final_mc_list, final_mc_entropy_list, final_obs_list, final_act_list = get_mc_return(bias_eval_env, 
                                                                                        agent, 
                                                                                        max_ep_len, 
                                                                                        alpha, 
                                                                                        gamma, 
                                                                                        n_mc_eval, 
                                                                                        n_mc_cutoff)
    logger.store(MCDisRet=final_mc_list)
    logger.store(MCDisRetEnt=final_mc_entropy_list)
    obs_tensor = Tensor(final_obs_list).to(agent.device)
    acts_tensor = Tensor(final_act_list).to(agent.device)
    with torch.no_grad():
        q_prediction = \
            agent.get_ave_q_prediction_for_bias_evaluation(obs_tensor, acts_tensor).cpu().numpy().reshape(-1)
    bias = q_prediction - final_mc_entropy_list
    bias_abs = np.abs(bias)
    bias_squared = bias ** 2
    logger.store(QPred=q_prediction)
    logger.store(QBias=bias)
    logger.store(QBiasAbs=bias_abs)
    logger.store(QBiasSqr=bias_squared)
    final_mc_entropy_list_normalize_base = final_mc_entropy_list.copy()
    final_mc_entropy_list_normalize_base = np.abs(final_mc_entropy_list_normalize_base)
    final_mc_entropy_list_normalize_base[final_mc_entropy_list_normalize_base < 10] = 10
    normalized_bias_per_state = bias / final_mc_entropy_list_normalize_base
    logger.store(NormQBias=normalized_bias_per_state)
    normalized_bias_sqr_per_state = bias_squared / final_mc_entropy_list_normalize_base
    logger.store(NormQBiasSqr=normalized_bias_sqr_per_state)

def main(env_name, seed=0, epochs='mbpo', steps_per_epoch=1000,
             max_ep_len=1000, n_evals_per_epoch=1,
             logger_kwargs=dict(), 
             hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
             lr=3e-4, gamma=0.99, polyak=0.995,
             alpha=0.2, auto_alpha=True, target_entropy='mbpo',
             start_steps=5000, delay_update_steps='auto',
             utd_ratio=20, num_Q=10, num_min=2, q_target_mode='min',
             policy_update_delay=20,
             n_mc_eval=1000, n_mc_cutoff=350, reseed_each_epoch=True
             ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if epochs == 'mbpo' or epochs < 0:
        epochs = mbpo_epoches[env_name]
    total_steps = steps_per_epoch * epochs + 1

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env_fn = lambda: gym.make(env_name)
    env, test_env, bias_eval_env = env_fn(), env_fn(), env_fn()

    torch.manual_seed(seed)
    np.random.seed(seed)
    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        test_env.seed(test_env_seed)
        test_env.action_space.np_random.seed(test_env_seed)
        bias_eval_env.seed(bias_eval_env_seed)
        bias_eval_env.action_space.np_random.seed(bias_eval_env_seed)
    seed_all(epoch=0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len
    act_limit = env.action_space.high[0].item()

    start_time = time.time()
    sys.stdout.flush()

    agent = Agent(env_name, obs_dim, act_dim, act_limit, device,
                 hidden_sizes, replay_size, batch_size,
                 lr, gamma, polyak,
                 alpha, auto_alpha, target_entropy,
                 start_steps, delay_update_steps,
                 utd_ratio, num_Q, num_min, q_target_mode,
                 policy_update_delay)

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    for t in range(total_steps):
        a = agent.get_exploration_action(o, env)
        o2, r, d, _ = env.step(a)

        ep_len += 1
        d = False if ep_len == max_ep_len else d

        agent.store_data(o, a, r, o2, d)
        agent.train(logger)

        o = o2
        ep_ret += r

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if (t+1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            test_agent(agent, test_env, max_ep_len, logger)
            log_bias_evaluation(bias_eval_env, agent, logger, \
                max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)

            # reseed should improve reproducibility 
            # (should make results the same whether bias evaluation is on or not)
            if reseed_each_epoch:
                seed_all(epoch)

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)
            logger.log_tabular("MCDisRet", with_min_and_max=True)
            logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
            logger.log_tabular("QPred", with_min_and_max=True)
            logger.log_tabular("QBias", with_min_and_max=True)
            logger.log_tabular("QBiasAbs", with_min_and_max=True)
            logger.log_tabular("NormQBias", with_min_and_max=True)
            logger.log_tabular("QBiasSqr", with_min_and_max=True)
            logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
            logger.dump_tabular()

            sys.stdout.flush()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=-1) # -1 means use mbpo epochs
    parser.add_argument('--exp_name', type=str, default='redq_sac')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--M', type=float, default=2)
    parser.add_argument('--G', type=int, default=20)
    parser.add_argument('--q_target_mode', type=str, default='min')
    args = parser.parse_args()

    exp_name_full = args.exp_name + '_n' + str(args.N) + '_m' + str(args.M) + '_g' + str(args.G) + '_q' + args.q_target_mode + '_%s' % args.env
    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)

    main(args.env, seed=args.seed, epochs=args.epochs,
             logger_kwargs=logger_kwargs, utd_ratio=args.G, num_Q=args.N, num_min=args.M, q_target_mode=args.q_target_mode)
    
