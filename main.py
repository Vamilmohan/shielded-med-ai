import torch
import json
import os

from src.data_loader import get_dataloaders
from src.model import CNNModel
from src.local_train import local_train
from src.federated_server import federated_average
from src.prototypes import build_class_prototypes
# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Limit CPU threads
torch.set_num_threads(1)
# =========================
# CONFIG
# =========================
NUM_CLIENTS = 3
ROUNDS = 3

TASK_CONFIG = {
    "chest": {
        "dataset": "Chest (Pneumonia)",
        "num_classes": 2
    },
    "path": {
        "dataset": "Pathology (Tumor)",
        "num_classes": 9
    },
    "skin": {
        "dataset": "Skin (Dermatology)",
        "num_classes": 7
    }
}

TASK_NAME = "skin"   # 🔁 change chest / path / skin

# =========================
# TRAINING
# =========================
def main():
    cfg = TASK_CONFIG[TASK_NAME]

    client_loaders, in_channels = get_dataloaders(
        cfg["dataset"],
        NUM_CLIENTS
    )

    global_model = CNNModel(
        num_classes=cfg["num_classes"],
        in_channels=in_channels
    )

    metrics = {
        "accuracy": [],
        "loss": [],
        "hospital_contribution": []
    }

    for r in range(ROUNDS):
        print(f"\n🔁 Federated Round {r+1}")

        local_state_dicts = []
        round_acc = []
        round_loss = []

        for i in range(NUM_CLIENTS):
            print(f"🏥 Training Hospital {i+1}")

            local_model = CNNModel(
                num_classes=cfg["num_classes"],
                in_channels=in_channels
            )
            local_model.load_state_dict(global_model.state_dict())

            state_dict, acc, loss = local_train(
                local_model,
                client_loaders[i]
            )

            local_state_dicts.append(state_dict)
            round_acc.append(acc)
            round_loss.append(loss)

        global_weights = federated_average(local_state_dicts)
        global_model.load_state_dict(global_weights)

        metrics["accuracy"].append(sum(round_acc) / NUM_CLIENTS)
        metrics["loss"].append(sum(round_loss) / NUM_CLIENTS)
        metrics["hospital_contribution"].append(round_acc)

    # =========================
    # SAVE
    # =========================
    save_dir = f"models/{TASK_NAME}"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        global_model.state_dict(),
        f"{save_dir}/global_model.pth"
    )

    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(metrics, f)

    print("✅ Federated Training Completed")
    global_model.eval()
    print("🔬 Building class prototypes")
    train_loader = client_loaders[0]  # just one is enough
    prototypes = build_class_prototypes(
        global_model,
        train_loader,
        cfg["num_classes"]
    )
    torch.save(
        prototypes,
        f"models/{TASK_NAME}/prototypes.pth"
    )


if __name__ == "__main__":
    main()