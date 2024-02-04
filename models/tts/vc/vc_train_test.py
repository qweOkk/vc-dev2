import argparse
import torch
from models.tts.vc.vc_trainer import VCTrainer
from utils.util import load_config


def build_trainer(args, cfg):
    supported_trainer = {
        "VC": VCTrainer,
    }

    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg)
    return trainer

def cuda_relevant(deterministic=False):
    torch.cuda.empty_cache()
    # TF32 on Ampere and above
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    # Deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="testVC",
        help="A specific name to note the experiment",
    )
    parser.add_argument(
        "--resume", action="store_true", help="The model name to restore"
    )
    parser.add_argument(
        "--log_level", default="warning", help="logging level (debug, info, warning)"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)


    # CUDA settings
    cuda_relevant()

    # Build trainer
    print("Building Trainer")
    trainer = build_trainer(args, cfg)
    print("Trainer Built")

    trainer.train_loop()


if __name__ == "__main__":
    main()