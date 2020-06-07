from trainer import Trainer
from trainer_clipped import Trainer_Clipped
from agent import Agent
from agent_clipped import Agent_Clipped
from env import FlappyBird
import torch
import argparse

parser = argparse.ArgumentParser(description='Flappy-bird Configuration')
parser.add_argument('--mode', dest='mode', default='train', type=str, help='[eval, train]')
parser.add_argument('--ckpt', dest='ckpt', default='none', type=str, help='[model_{}.pth.tar]')
parser.add_argument('--cuda', dest='cuda', default='Y', type=str, help='[Y/N]')
parser.add_argument('--clipped', dest='clipped', default=True, type=bool, help='[True, False]')

# Apenas foi acrescentado clipped nas variaveis necessárias

if __name__ == '__main__':
    args = parser.parse_args()
    use_gpu = (args.cuda == 'Y')
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    if args.clipped == True:
        agent_clipped = Agent_Clipped(cuda=torch.cuda.is_available() if use_gpu else 'cpu')
        if args.mode == 'train':
            env = FlappyBird(record_every_episode=100, outdir='tmp/result/')
            tr = Trainer_Clipped(agent_clipped, env)
            if args.ckpt != 'none':
                tr.load(args.ckpt, device)
            tr.train(device=device)
        else:
            env = FlappyBird(record_every_episode=1, outdir='eval/')
            tr = Trainer_Clipped(agent_clipped, env)
            tr.load(args.ckpt, device)
            accumulated_reward, step = tr.run(device=device, explore=False)
            print('Accumulated_reward: {}, alive time: {}'.format(accumulated_reward, step))
    else:
        agent = Agent(cuda=torch.cuda.is_available() if use_gpu else 'cpu')
        if args.mode == 'train':
            env = FlappyBird(record_every_episode=100, outdir='tmp/result/')
            tr = Trainer(agent, env)
            if args.ckpt != 'none':
                tr.load(args.ckpt, device)
            tr.train(device=device)
        else:
            env = FlappyBird(record_every_episode=1, outdir='eval/')
            tr = Trainer(agent, env)
            tr.load(args.ckpt, device)
            accumulated_reward, step = tr.run(device=device, explore=False)
            print('Accumulated_reward: {}, alive time: {}'.format(accumulated_reward, step))
