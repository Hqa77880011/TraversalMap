"""Trainable shared mask-crop encoder for class, severity and association features."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image


def read_manifest(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Crop manifest is empty")
    return rows, path.parent


class Crops(Dataset):
    def __init__(self, manifest, configuration, supervised=True):
        self.rows, self.root = read_manifest(manifest)
        self.configuration, self.supervised = configuration, supervised

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, size = self.rows[index], self.configuration.get("image_size", 64)
        with Image.open(self.root / row["image"]) as source:
            image = np.array(source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)/255
        if "mask" in row:
            with Image.open(self.root / row["mask"]) as source:
                mask = np.asarray(source.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
        else:
            mask = np.ones((size, size), dtype=bool)
        if not mask.any():
            raise ValueError("A semantic crop requires a nonempty supplied mask")
        value = torch.from_numpy(np.concatenate([image.transpose(2, 0, 1), mask[None].astype(np.float32)]))
        result = dict(x=value, index=index)
        if self.supervised:
            result.update(class_target=self.configuration["class_labels"].index(row["class"]),
                          severity_target=self.configuration["severity_levels"].index(row["severity"]))
        return result


class SemanticNetwork(nn.Module):
    """One small shared CNN; three outputs use the same masked representation."""

    def __init__(self, configuration):
        super().__init__()
        hidden = configuration.get("hidden", 32)
        self.encoder = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, hidden, 3, stride=2, padding=1), nn.ReLU())
        self.class_head = nn.Linear(hidden, len(configuration["class_labels"]))
        self.severity_head = nn.Linear(hidden, len(configuration["severity_levels"]))

    def forward(self, images):
        feature = self.encoder(images)
        mask = F.interpolate(images[:, 3:4], size=feature.shape[-2:], mode="nearest")
        pooled = (feature*mask).sum((2, 3))/mask.sum((2, 3)).clamp_min(1)
        return dict(class_logits=self.class_head(pooled), severity_logits=self.severity_head(pooled),
                    descriptor=F.normalize(pooled, dim=-1))


def loss(prediction, batch):
    return F.cross_entropy(prediction["class_logits"], batch["class_target"])+F.cross_entropy(
        prediction["severity_logits"], batch["severity_target"])


def train(train_manifest, validation_manifest, output, configuration=None):
    """Train on source-pipe groups and select only by validation class/severity CE."""
    training_rows, _ = read_manifest(train_manifest)
    validation_rows, _ = read_manifest(validation_manifest)
    if any("group" not in row for row in training_rows+validation_rows):
        raise ValueError("Every supervised crop needs its physical pipe/segment group")
    if {row["group"] for row in training_rows} & {row["group"] for row in validation_rows}:
        raise ValueError("Training and validation pipe groups overlap")
    cfg = dict(seed=13, epochs=150, patience=15, batch_size=16, hidden=32, image_size=64, learning_rate=1e-3)
    cfg.update(configuration or {})
    cfg.setdefault("class_labels", sorted({row["class"] for row in training_rows}))
    cfg.setdefault("severity_levels", sorted({row["severity"] for row in training_rows}))
    if min(cfg["epochs"], cfg["batch_size"]) < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    model = SemanticNetwork(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    training = DataLoader(Crops(train_manifest, cfg), batch_size=cfg["batch_size"], shuffle=True,
                          generator=torch.Generator().manual_seed(cfg["seed"]))
    validation = DataLoader(Crops(validation_manifest, cfg), batch_size=cfg["batch_size"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    best, stale = float("inf"), 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in training:
            optimizer.zero_grad()
            value = loss(model(batch["x"]), batch)
            if not torch.isfinite(value):
                raise FloatingPointError("Nonfinite semantic training loss")
            value.backward(); optimizer.step()
        model.eval()
        total, count = 0., 0
        with torch.no_grad():
            for batch in validation:
                total += float(loss(model(batch["x"]), batch))*len(batch["x"])
                count += len(batch["x"])
        valid = total/count
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(epoch=epoch+1, validation_loss=valid))+"\n")
        if valid < best:
            best, stale = valid, 0
            torch.save(dict(configuration=cfg, state=model.state_dict()), output / "model.pt")
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    return output / "model.pt"


def predict(manifest, checkpoint):
    """Read real images/masks and export native learned likelihoods and descriptors."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    configuration = payload["configuration"]
    model = SemanticNetwork(configuration).eval()
    model.load_state_dict(payload["state"])
    dataset = Crops(manifest, configuration, supervised=False)
    rows = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=configuration["batch_size"]):
            result = model(batch["x"])
            for index, source in enumerate(batch["index"]):
                row = dataset.rows[int(source)]
                rows.append(dict(id=row["id"], class_labels=configuration["class_labels"],
                    severity_levels=configuration["severity_levels"],
                    class_probabilities=result["class_logits"][index].softmax(-1).tolist(),
                    severity_probabilities=result["severity_logits"][index].softmax(-1).tolist(),
                    descriptor=result["descriptor"][index].tolist(), quality=row.get("quality", 1.),
                    source_frame=row.get("source_frame", row["image"])))
    return rows


def attach(episodes, evidence):
    lookup = {row["id"]: row for row in evidence}
    if len(lookup) != len(evidence):
        raise ValueError("Semantic prediction IDs must be unique")
    for episode in episodes:
        for r, traversal in enumerate(episode["traversals"]):
            for k, observation in enumerate(traversal["observations"]):
                identity = observation.get("id", f"{episode['id']}:{r}:{k}")
                value = lookup[identity]
                for key in ("class_labels", "severity_levels"):
                    if key in episode and episode[key] != value[key]:
                        raise ValueError("Episode and semantic checkpoint label orders differ")
                    episode[key] = value[key]
                observation.update({key: value[key] for key in ("class_probabilities", "severity_probabilities",
                                                               "descriptor", "quality", "source_frame")})
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("train")
    item.add_argument("--train", required=True); item.add_argument("--validation", required=True)
    item.add_argument("--config"); item.add_argument("--output", required=True)
    item = sub.add_parser("predict")
    item.add_argument("--manifest", required=True); item.add_argument("--checkpoint", required=True)
    item.add_argument("--output", required=True)
    item = sub.add_parser("attach")
    item.add_argument("--episodes", required=True); item.add_argument("--evidence", required=True)
    item.add_argument("--output", required=True)
    args = parser.parse_args()
    from .cli import read_jsonl, write_jsonl
    if args.command == "train":
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
        train(args.train, args.validation, args.output, cfg)
    elif args.command == "predict":
        write_jsonl(args.output, predict(args.manifest, args.checkpoint))
    else:
        write_jsonl(args.output, attach(read_jsonl(args.episodes), read_jsonl(args.evidence)))


if __name__ == "__main__":
    main()
