import json

import hydra
import wandb
from omegaconf import OmegaConf

from utils.wandb_hydra import wandb_init


def get_old_cfg():
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.ENV.WALKER_DIR = f"{'/'.join(__file__.split('/')[:-1])}/{cfg.ENV.WALKER_DIR.replace('./', '')}"
    cfg.UNIMAL_TEMPLATE = f"{'/'.join(__file__.split('/')[:-1])}/{cfg.UNIMAL_TEMPLATE.replace('./', '')}"

    # Set cfg options which are inferred
    set_cfg_options()

    with open("./oldcfg.json", "w") as f:
        f.write(json.dumps(cfg.asdict(), indent=2))
    exit()

@hydra.main(version_base=None, config_path="config", config_name="ft")
def main(cfg):
    #get_old_cfg()



    #import metamorph.config as tmp
    #tmp.cfg = cfg

    cfg.ENV.WALKER_DIR = f"{cfg.UNIMAL_PATH}/{cfg.ENV.WALKER_SUBDIR}"
    #override_cfg(cfg)


    from metamorph.config import set_cfg  # , override_cfg  # set_cfg,
    set_cfg(cfg)
    from tools.train_ppo import set_cfg_options, ppo_train, parse_args
    set_cfg_options()

    print("=== WANDB INIT ===")
    wandb_init(cfg)
    print("DONE")

    print("=== Training begins ===")
    ppo_train()


    with open("./newcfg.json", "w") as f:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        json_config = json.dumps(cfg_dict, indent=2)
        f.write(json_config)

    exit()


    # Save the config
    ppo_train()

    print(cfg)
    pass

if __name__ == "__main__":
    import sys
    #sys.argv = ["main.py", "--cfg", "/home/charlie/Desktop/metamorph/configs/ft.yaml"]
    #get_old_cfg()
    main()