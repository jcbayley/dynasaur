import argparse
from dynasaur.config import read_config
import os
from dynasaur.train_model import run_training, run_testing
from dynasaur.train_model_timestep import run_training as run_training_timestep
from dynasaur.train_model_timestep import run_testing as run_testing_timestep



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False, default="none")
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--continuetrain', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--makeplots', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ntest", type=int, required=False, default=10)
    args = parser.parse_args()

    config = read_config(os.path.abspath(args.config))

    continue_train = args.continuetrain
    train_model = args.train
    test_model = args.test
    print("makeplots", args.makeplots)
        
    if train_model:
        if config.get("Data", "timestep-predict"):
            run_training_timestep(config, continue_train=continue_train)
        else:
            run_training(config, continue_train=continue_train)

    if test_model:
        if config.get("Data", "timestep-predict"):
            run_testing_timestep(config, n_test=args.ntest, make_plots=args.makeplots)
        else:
            run_testing(config, n_test=args.ntest, make_plots=args.makeplots)
