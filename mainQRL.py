from RecordCandle import ReadCandles
from Lob import ReadLobs
from Env import QRLEnv
from Model import DQN

import torch
import gym
import os
import ptan

import torch.nn as nn
import torch.optim as optim
import sys
import time
import numpy as np

DEFAULT_EVENT_COUNT = 100
DEFAULT_TRADE_COUNT = 10
DEFAULT_CANDLE_COUNT = 10
DEFAULT_LOB_NUM_LEVEL = 10
DEFAULT_EVENT_WIDTH = 7
DEFAULT_TRADE_WIDTH = 5
DEFAULT_CANDLE_WIDTH = 4
DEFAULT_AP_WIDTH = 5
DEFAULT_SUPRES_WIDTH = 6
DEFAULT_ACTION_N = 5

BATCH_SIZE = 32
BARS_COUNT = 500
TARGET_NET_SYNC = 1000

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000

REWARD_STEPS = 1

LEARNING_RATE = 0.0001

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000

EPSILON_START = 1.0
EPSILON_STOP = 0.1
EPSILON_STEPS = 1000000

CHECKPOINT_EVERY_STEP = 100000
VALIDATION_EVERY_STEP = 100000


class RewardTracker:
    def __init__(self, stop_reward, group_rewards=1):
        self.stop_reward = stop_reward
        self.reward_buf = []
        self.steps_buf = []
        self.group_rewards = group_rewards

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        self.total_steps = []
        return self

    def __exit__(self, *args):
        pass

    def reward(self, reward_steps, frame, epsilon=None):
        reward, steps = reward_steps
        self.reward_buf.append(reward)
        self.steps_buf.append(steps)
        if len(self.reward_buf) < self.group_rewards:
            return False
        reward = np.mean(self.reward_buf)
        steps = np.mean(self.steps_buf)
        self.reward_buf.clear()
        self.steps_buf.clear()
        self.total_rewards.append(reward)
        self.total_steps.append(steps)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        mean_steps = np.mean(self.total_steps[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, mean steps %.2f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards)*self.group_rewards, mean_reward, mean_steps, speed, epsilon_str
        ))
        sys.stdout.flush()

        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_actions = net(next_states_v).max(1)[1]
    next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


# initialize networks
if True:
    cuda = True
    run = 'qrl'
    device = torch.device("cuda" if cuda else "cpu")
    
    saves_path = os.path.join("/home/user/QRLDumps", run)
    os.makedirs(saves_path, exist_ok=True)

    # read big array
    lob_order, lobs = ReadLobs('lob.txt')
    candles15min_order, candles15min = ReadCandles('min15.txt')
    candles1min_order, candles1min = ReadCandles('min1.txt')
    candles1sec_order, candles1sec = ReadCandles('sec1.txt')

    # find 4h point after start and 15 min point before end of data
    lob_order_len = len(lob_order)
    start_time = lob_order[0]
    ind = 0
    cur_time = lob_order[ind]
    diff = 4 * 60 * 60 * 1000  # 4 hours in ms
    
    while ((cur_time - start_time) < diff):
        ind += 1
        cur_time = lob_order[ind]
        
    start_random_time = ind
    
    # find 1h point before end of data
    end_time = lob_order[lob_order_len - 1]
    ind = (lob_order_len - 1)
    cur_time = lob_order[ind]
    diff = 60 * 60 * 1000  # 1h im ms
    
    while ((end_time - cur_time) < diff):
        ind -= 1
        cur_time = lob_order[ind]
    
    end_random_time = ind    
    
    env = QRLEnv(lob_order, lobs, candles15min_order, candles15min, candles1min_order, candles1min, candles1sec_order, candles1sec, start_random_time, end_random_time, DEFAULT_EVENT_COUNT, DEFAULT_TRADE_COUNT, DEFAULT_CANDLE_COUNT)
    env.reset()           
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)  

    print(env.observation_space.shape)
    print(env.action_space.n)
    
    net = DQN(DEFAULT_LOB_NUM_LEVEL, DEFAULT_EVENT_COUNT, DEFAULT_EVENT_WIDTH, DEFAULT_CANDLE_COUNT, DEFAULT_CANDLE_WIDTH, 
              DEFAULT_TRADE_COUNT, DEFAULT_TRADE_WIDTH, DEFAULT_AP_WIDTH, DEFAULT_SUPRES_WIDTH, DEFAULT_ACTION_N).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    # main training loop
    step_idx = 0
    eval_states = None
    best_mean_val = None
    
    
# do the main learning loop 
if True:
    with  RewardTracker(np.inf, group_rewards=100) as reward_tracker:
        while True:
            step_idx += 1
            buffer.populate(1)
            selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)

            new_rewards = exp_source.pop_rewards_steps()
            if new_rewards:
                val = reward_tracker.reward(new_rewards[0], step_idx, selector.epsilon)
                if val:
                  break

            if len(buffer) < REPLAY_INITIAL:
                continue

            if eval_states is None:
                print("Initial buffer populated, start training")
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            if step_idx % EVAL_EVERY_STEP == 0:
                mean_val = calc_values_of_states(eval_states, net, device=device)
                #writer.add_scalar("values_mean", mean_val, step_idx)
                if best_mean_val is None or best_mean_val < mean_val:
                    if best_mean_val is not None:
                        print("%d: Best mean value updated %.3f -> %.3f" % (step_idx, best_mean_val, mean_val))
                    best_mean_val = mean_val                   

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_v = calc_loss(batch, net, tgt_net.target_model, GAMMA ** REWARD_STEPS, device=device)
            loss_v.backward()
            optimizer.step()

            if step_idx % TARGET_NET_SYNC == 0:
                tgt_net.sync()

            if step_idx % CHECKPOINT_EVERY_STEP == 0:
                idx = step_idx // CHECKPOINT_EVERY_STEP
                torch.save(net.state_dict(), os.path.join(saves_path, "checkpoint-%3d.data" % idx))    
    
    



  


    
    
        
        
        

