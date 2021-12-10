import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description="PyTorch DQN Training")

    parser.add_argument("--savePrefix", type=str, help="The directory prefix for saving checkpoints")
    parser.add_argument("--epsStart", type=float, help="The initial epsilon")
    parser.add_argument("--epsEnd", type=float, help="The final epsilon")
    parser.add_argument("--epsDecay", type=float, help="The total epsilon decay steps")
    parser.add_argument("--rlmodel", type=str, default="DQN", choices=["DQN"], help="Model to use")
    parser.add_argument("--restore", type=str, default="None", help="Pretrained model to resume")
    args = parser.parse_args()

    return args