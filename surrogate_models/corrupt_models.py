import os
from nequip.scripts.deploy import load_deployed_model


import numpy as np
from pathlib import Path
import torch
import argparse
import shutil

if __name__ == "__main__":
    corrpution_facs = np.logspace(-4.3,-0.3,12)
    cdiff = np.diff(corrpution_facs)
    noise_arrs = []
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--seed", type=int, required=True, help="Seed to corrupt model")
    parser.add_argument(
        "--modelname", type=str, required=True, help="Seed to corrupt model"
    )
    args = parser.parse_args()
    SEED = args.seed
    DEVICE = "cpu"
    torch.manual_seed(SEED)
    base_model_loc = Path("./")

    base_model_name = args.modelname
    base_model_file = base_model_loc / base_model_name
    tmp = base_model_name.split(".")[0]
    corrupted_model_dir = base_model_loc / f"{tmp}_corrupted_SEED_{SEED}"
    if corrupted_model_dir.exists():
        shutil.rmtree(corrupted_model_dir)
    corrupted_model_dir.mkdir()

    model, metadata_dict = load_deployed_model(
        model_path=base_model_file, device=DEVICE, freeze=False
    )

    for param in model.parameters():
        param_std = torch.std(param)
        noise_arrs.append(
            torch.normal(0.0, param_std.item(), param.shape, device=DEVICE)
        )

    for i in range(0, corrpution_facs.shape[0]):
        cfac_file = corrupted_model_dir / f"corruptfac_{corrpution_facs[i]}_{i}.pth"
        if cfac_file.exists():
            os.remove(cfac_file)

        if i == 0:
            add_fac = corrpution_facs[0]
        else:
            add_fac = cdiff[i - 1]
        for ct, param in enumerate(model.parameters()):
            param.data = param + add_fac * noise_arrs[ct]
            param_std = torch.std(param.data)
            if ct == 0:
                print(
                    f"{corrpution_facs[i], param_std.item(), torch.std(add_fac * noise_arrs[ct]).item()}"
                )

        torch.jit.save(model, cfac_file, _extra_files=metadata_dict)