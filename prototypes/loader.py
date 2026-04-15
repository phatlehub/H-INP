import torch


def load_prior_bank(prior_path: str, device: str = "cpu", dtype=None) -> torch.Tensor:
    """Load prior prototype bank from disk and move to target device."""
    prior_obj = torch.load(prior_path, map_location="cpu")

    if isinstance(prior_obj, dict):
        if "prior_bank" in prior_obj:
            prior_bank = prior_obj["prior_bank"]
        elif "prototypes" in prior_obj:
            prior_bank = prior_obj["prototypes"]
        else:
            raise KeyError("Cannot find prior bank in checkpoint. Expected keys: 'prior_bank' or 'prototypes'.")
    else:
        prior_bank = prior_obj

    if not isinstance(prior_bank, torch.Tensor):
        prior_bank = torch.tensor(prior_bank)

    prior_bank = prior_bank.float()
    if dtype is not None:
        prior_bank = prior_bank.to(dtype=dtype)
    return prior_bank.to(device)
