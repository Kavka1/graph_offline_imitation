import argparse
import os
import subprocess
from pathlib import Path

from graph_offline_imitation.utils.config import Config


WANDB_API_KEY = '65024cebe906bccf53abc001e0b8f6ca2c2cbd5b'


def try_wandb_setup(path, config):
    # wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_api_key = WANDB_API_KEY
    if wandb_api_key is not None and wandb_api_key != "":
        try:
            import wandb
        except:
            return
        # project_dir = os.path.dirname(os.path.dirname(__file__))
        split_path          =   path.split('/')
        env_name_as_group   =   split_path[-3]
        alg_name_as_proj    =   split_path[-2]
        alg_name_wo_seed    =   alg_name_as_proj.split('-')[0]
        wb = wandb.init(
            project =   'OIL',
            group   =   f"{env_name_as_group}-{alg_name_wo_seed}",
            name    =   f"{env_name_as_group}-{alg_name_as_proj}",
            reinit  =   True,
            id      =   f"{env_name_as_group}-{alg_name_as_proj}",
            dir     =   path,
            # config  =   config
        )
        # wandb.init(
        #     project=os.path.basename(project_dir),
        #     name=os.path.basename(path),
        #     config=config.flatten(separator="-"),
        #     dir=os.path.join(os.path.dirname(project_dir), "wandb"),
        # )
        return wb
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default="oi_antmaze_umaze_exp_div_1/bc_all")
    parser.add_argument('--notation', '-n', type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--seeds", '-s', type=int, nargs='+', default=[8])
    args     = parser.parse_args()

    pth         = args.path    
    args.config = f"{Path(__file__).resolve().parent}/../configs/{pth}.yaml"
    args.path   = f"{Path(__file__).resolve().parent}/../results/{pth}/"

    org_path    = args.path

    for seed in args.seeds:
        # add seed to path
        if args.notation:
            args.path = org_path[:-1] + f'-{seed}-{args.notation}/'
        else:
            args.path = org_path[:-1] + f'-{seed}/'
        config = Config.load(args.config)
        # update config with result path and seed
        config.update(dict(seed=seed, res_path=args.path))
        os.makedirs(args.path, exist_ok=True)
        wb = try_wandb_setup(args.path, config)
        config.save(args.path)  # Save the config
        # save the git hash
        process = subprocess.Popen(["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE)
        git_head_hash = process.communicate()[0].strip()
        with open(os.path.join(args.path, "git_hash.txt"), "wb") as f:
            f.write(git_head_hash)
        # Parse the config file to resolve names.
        config = config.parse()
        # Get the model
        model = config.get_model(device=args.device)
        # Get the trainer
        trainer = config.get_trainer()
        # Train the model
        trainer.set_model(model)
        trainer.train(args.path, wb)
