import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic

import gym
from tensorboardX import SummaryWriter

def test(rank, args, shared_model, counter, num_done, num_episode, reward_, lock):
    torch.manual_seed(args.seed + rank)


    env = gym.make('MountainCar-v0').unwrapped

    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state).float()
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=200)
    episode_length = 0
    episode = 0 
    #while episode < 24000000000:
    while counter.value < 120000000 :
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 4).float()
            hx = torch.zeros(1, 4).float()
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            #print(value)
            #print(logit)
            #print(state)
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0].item())
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            reward_.value = int(reward_sum)
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            # arr.value.append(num_done.value / num_episode.value)
            with lock:
                num_episode.value = 0
                num_done.value = 0
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(10)

        state = torch.from_numpy(state).float()
