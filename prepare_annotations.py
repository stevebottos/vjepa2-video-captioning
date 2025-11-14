import json
from pathlib import Path
from tqdm import tqdm

annotations_path = "/media/steve/storage/20bn-something-something-v2-labels"

annotations_train = json.loads((Path(annotations_path) / "train.json").read_text())
train_paraphrases = json.loads(Path("ssv2_paraphrase.json").read_text())

for ann in tqdm(annotations_train):
    del ann["template"]
    if ann["id"] in train_paraphrases:
        ann["label"] = [ann["label"]] + train_paraphrases[ann["id"]]
    else:
        ann["label"] = [ann["label"]]

with open(Path(annotations_path) / "train_ready.json", "w") as f:
    json.dump(str(annotations_train), f)

annotations_val = json.loads((Path(annotations_path) / "validation.json").read_text())
for ann in tqdm(annotations_val):
    del ann["template"]
    ann["label"] = [ann["label"]]

with open(Path(annotations_path) / "val_ready.json", "w") as f:
    json.dump(str(annotations_val), f)
