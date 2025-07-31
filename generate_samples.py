import torch
import tqdm
from model import OneHotCVAE
import torch.nn.functional as F
import os
import argparse
from torchvision.utils import save_image
import pickle  # for exception type


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_folder", type=str, required=True, help="Path to folder of VAE"
    )

    parser.add_argument(
        "--n_samples", type=int, default=1000, help="Number of samples to generate"
    )

    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size. Keep it so that n_samples is divisible by batch_size."
    )

    parser.add_argument(
        "--label_to_generate", type=int, default=0, help="Which MNIST class to generate"
    )

    args = parser.parse_args()
    assert args.n_samples % args.batch_size == 0, "Ensure n_samples is a multiple of batch_size!"
    return args


def load_checkpoint(ckpt_path, device):
    """
    Load checkpoint, handling the PyTorch 2.6+ weights_only behavior and allowing
    argparse.Namespace if needed (only if the checkpoint is trusted).
    """
    try:
        # First try the default load (may use weights_only=True internally)
        return torch.load(ckpt_path, map_location=device)
    except pickle.UnpicklingError:
        # Likely blocked due to argparse.Namespace; allowlist it explicitly.
        with torch.serialization.safe_globals([argparse.Namespace]):
            # Force full load (weights_only=False) to get the saved Namespace.
            return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch might not accept weights_only arg; fallback.
        return torch.load(ckpt_path, map_location=device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    ckpt_path = os.path.join(args.ckpt_folder, "ckpts/ckpt.pt")
    ckpt = load_checkpoint(ckpt_path, device)

    # normalize config to dict
    config = ckpt.get("config", {})
    if isinstance(config, argparse.Namespace):
        config = vars(config)

    # build model
    vae = OneHotCVAE(
        x_dim=config["x_dim"],
        h_dim1=config["h_dim1"],
        h_dim2=config["h_dim2"],
        z_dim=config["z_dim"],
    )
    vae = vae.to(device)

    vae.load_state_dict(ckpt["model"])
    vae.eval()

    sample_dir = os.path.join(args.ckpt_folder, f"{args.label_to_generate}_samples")
    os.makedirs(sample_dir, exist_ok=True)

    i = 0
    with torch.no_grad():
        for _ in tqdm.tqdm(range(args.n_samples // args.batch_size)):
            z = torch.randn((args.batch_size, config["z_dim"])).to(device)
            c = (torch.ones(args.batch_size, dtype=int) * args.label_to_generate).to(device)
            c = F.one_hot(c, 10)
            samples = vae.decoder(z, c).view(-1, 1, 28, 28)
            for x in samples:
                save_image(x, os.path.join(sample_dir, f"{i}.png"))
                i += 1