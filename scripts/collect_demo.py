import argparse
import os


from graph_offline_imitation.utils.config import Config


def collect_demos():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to the model",
                        default= "/home/PJLAB/kang/proj/graph_offline_imitation/results/kitchen/gofar/best_model.pt"
                        )
    parser.add_argument("--path", type=str, help="Path to save the gif",
                        default='/home/PJLAB/kang/proj/graph_offline_imitation/demos/kitchen/gofar/', 
                        )
    parser.add_argument("--num-ep", type=int, help="Number of episodes",
                        default=100,
                        )
    parser.add_argument("--save_ep", type=bool, default=True, help="Number of gifs to save.")
    parser.add_argument("--save_gif", type=bool, default=True, help="Number of gifs to save.")
    parser.add_argument("--save_only_success", type=bool, default=True, help="Number of gifs to save.")

    parser.add_argument("--width", type=int, default=400, help="Width of image")
    parser.add_argument("--height", type=int, default=360, help="Height of image")

    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--every-n-frames", type=int, default=2, help="Save every n frames to the gif.")
    parser.add_argument("--strict", action="store_true", default=False, help="Strict")
    parser.add_argument(
        "--terminate-on-success", action="store_true", default=False, help="Terminate gif on success condition."
    )
    parser.add_argument(
        "--override",
        metavar="KEY=VALUE",
        nargs="+",
        default=[],
        help="Set kv pairs used as args for the entry point script.",
    )
    parser.add_argument("--max-len", type=int, default=1000, help="maximum length of an episode.")
    args = parser.parse_args()

    assert args.checkpoint.endswith(".pt"), "Must provide a model checkpoint"
    config = Config.load(os.path.dirname(args.checkpoint))
    config["checkpoint"] = None  # Set checkpoint to None

    # Overrides
    print("Overrides:")
    for override in args.override:
        print(override)

    # Overrides
    for override in args.override:
        items       = override.split("=")
        key, value  = items[0].strip(), "=".join(items[1:])
        # Progress down the config path (separated by '.') until we reach the final value to override.
        config_path = key.split(".")
        config_dict = config
        while len(config_path) > 1:
            config_dict = config_dict[config_path[0]]
            config_path.pop(0)
        config_dict[config_path[0]] = value

    if len(args.override) > 0:
        print(config)

    # Over-write the parameters in the eval_kwargs
    config["trainer_kwargs"]["eval_fn"]                             = "render_policy"

    config["trainer_kwargs"]["eval_kwargs"]["num_ep"]               = args.num_ep
    config["trainer_kwargs"]["eval_kwargs"]["save_ep"]              = args.save_ep
    config["trainer_kwargs"]["eval_kwargs"]["save_gif"]             = args.save_gif
    config["trainer_kwargs"]['eval_kwargs']['save_only_success']    = args.save_only_success

    config["trainer_kwargs"]["eval_kwargs"]["width"]                = args.width
    config["trainer_kwargs"]["eval_kwargs"]["height"]               = args.height
    config["trainer_kwargs"]["eval_kwargs"]["every_n_frames"]       = args.every_n_frames
    config["trainer_kwargs"]["eval_kwargs"]["terminate_on_success"] = args.terminate_on_success

    config['wrapper_kwargs']['test_fraction']                       = 1       # to collect diverse demos
    
    config      = config.parse()
    model       = config.get_model(device=args.device)
    metadata    = model.load(args.checkpoint)
    trainer     = config.get_trainer()
    trainer.set_model(model)
    # Run the evaluation loop
    os.makedirs(args.path, exist_ok=True)
    metrics     = trainer.evaluate(args.path, metadata["current_step"])

    print("[research] Eval policy finished:")
    for k, v in metrics.items():
        print(k, v)



if __name__ == "__main__":
    collect_demos()