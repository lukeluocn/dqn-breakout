import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch DQN Training')

    parser.add_argument('--savePrefix', type=str, help='the save directory of the running results')
    parser.add_argument('--epsStart', type=float, help='the initial epsilon')
    parser.add_argument('--rlmodel', type=str, default='None', choices=['DQN', 'DuelingDQN'], help='model of experiment (default: DQN)')
    parser.add_argument('--restore', type=str, default='None', help='path to pretrained/resume model (default: None)')
    args = parser.parse_args()

    return args