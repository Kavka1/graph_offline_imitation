import argparse
import os
import subprocess
from typing import Tuple

from graph_offline_imitation.utils.config import OnlineConfig
from graph_offline_imitation.envs         import NEW_REGISTERED_ENV
from graph_offline_imitation.algs         import VALID_ALGORITHMS


def try_wandb_setup(path, config) -> None:
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is not None and wandb_api_key != "":
        try:
            import wandb
        except:
            return
        project_dir = os.path.dirname(os.path.dirname(__file__))
        wandb.init(
            project=os.path.basename(project_dir),
            name=os.path.basename(path),
            config=config.flatten(separator="-"),
            dir=os.path.join(os.path.dirname(project_dir), "wandb"),
        )


def obtain_config_result_path(alg: str, env: str, seed: int, prefix: str = None) -> Tuple[str]:
    suit_task = env.split('+')
    suit, task= suit_task[0], suit_task[1]
    assert alg in VALID_ALGORITHMS
    if suit in NEW_REGISTERED_ENV:
        assert task in NEW_REGISTERED_ENV[suit]
        print(f' - - - Running on new registered env {suit_task} - - - ')
    config = f"/home/PJLAB/kang/proj/graph_offline_imitation/configs/{suit}/{alg}.yaml"
    if prefix is not None:
        path   = f"/home/PJLAB/kang/proj/graph_offline_imitation/results/{suit}+{task}/{alg}+{seed}+{prefix}/"
    else:
        path   = f"/home/PJLAB/kang/proj/graph_offline_imitation/results/{suit}+{task}/{alg}+{seed}/"
    return suit, task, config, path


def launch(alg: str, env: str, seed: int, device: str = 'auto', prefix: str = None) -> None:
    ## Preprocess args
    suit, task, config_path, result_path = obtain_config_result_path(alg, env, seed, prefix)
    
    ## Prepare config
    config = OnlineConfig.load(config_path)
    # update seed & env in config
    config.update({'env': task, 'seed': seed, 'res_path': result_path})
    os.makedirs(result_path, exist_ok=True)
    try_wandb_setup(result_path, config)
    config.save(result_path)  # Save the config
    # save the git hash
    process         = subprocess.Popen(["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE)
    git_head_hash   = process.communicate()[0].strip()
    with open(os.path.join(result_path, "git_hash.txt"), "wb") as f:
        f.write(git_head_hash)

    ## Parse the config file to resolve names.
    config  = config.parse()
    # Get the model
    model   = config.get_model(device=device)
    # Get the trainer
    trainer = config.get_trainer()
    # # Train the model
    trainer.set_model(model)
    trainer.train(result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', '-a', type=str, default='goplanv3')
    parser.add_argument('--env', '-e', type=str, default='kitchen+kitchen-kettle_microwave_bottomburner_hinge-v0')
    parser.add_argument('--seed', '-s', type=int, default=10)
    parser.add_argument('--prefix', '-p', type=str, default=None)
    parser.add_argument('--device', '-d', type=str, default='auto')
    args   = parser.parse_args()
    launch(args.alg, args.env, args.seed, args.device, args.prefix)