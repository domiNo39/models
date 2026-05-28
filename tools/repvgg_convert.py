import argparse
import torch
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, Logger

def main(args):
    load_config(cfg, args.config)
    model = build_model(cfg.model)
    checkpoint = torch.load(args.model, map_location='cpu')
    if "state_dict" in checkpoint:
        raw_state_dict = checkpoint["state_dict"]
    else:
        raw_state_dict = checkpoint

    has_ema = any(k.startswith("avg_model.") for k in raw_state_dict.keys())
    new_state_dict = {}
    for k, v in raw_state_dict.items():
        if has_ema:
            if k.startswith("avg_model."):
                clean_name = k[10:]
                new_state_dict[clean_name] = v
        else:
            if k.startswith("model."):
                clean_name = k[6:]
                new_state_dict[clean_name] = v
            else:
                new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    if hasattr(model.backbone, "switch_to_deploy"):
        model.backbone.switch_to_deploy()
      else:
        for module in model.backbone.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        if hasattr(model.backbone, 'deploy'):
            model.backbone.deploy = True

    torch.save({"state_dict": model.state_dict()}, args.save_path)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert RepVGG trained model to inference model"
    )
    parser.add_argument("--config", help="path to config file", required=True)
    parser.add_argument("--model", help="path to trained .pth model", required=True)
    parser.add_argument("--save_path", help="path to save converted model", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)