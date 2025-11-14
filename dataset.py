from pathlib import Path
from torch.utils.data import Dataset
import torch
import random

random.seed(42)


class CaptionedClipsPreprocessed(Dataset):
    """
    Dataset for loading preprocessed V-JEPA features from PyTorch tensor files.
    Much faster and more memory efficient than loading videos at training time.
    """

    def __init__(
        self,
        preprocessed_root_path,
        preprocessed_root_path_2,
        annotations,
        split="train",
    ):
        """
        Args:
            preprocessed_root_path: Path to preprocessed directory (e.g., /media/steve/storage/densevid_preprocessed)
            split: "train" or "val"
            preprocessed_root_path_2: Optional second path to load additional preprocessed files from
            annotations_path: Path to annotations directory (e.g., /media/steve/storage/20bn-something-something-v2-labels)
        """
        # Collect paths from both root directories
        root_paths = [Path(preprocessed_root_path)]
        if preprocessed_root_path_2 is not None:
            root_paths.append(Path(preprocessed_root_path_2))

        # Find all PT files from all root paths
        self.sample_files = []
        for root_path in root_paths:
            split_dir = root_path / split

            if split_dir.exists():
                files = sorted(split_dir.glob("*.pt"))
                self.sample_files.extend(files)
                print(f"Found {len(files)} preprocessed {split} samples in {split_dir}")

        if len(self.sample_files) == 0:
            raise ValueError(
                f"No preprocessed samples found in {[str(p / split) for p in root_paths]}\n"
                f"Please run preprocess.py first to generate preprocessed features."
            )

        print(f"Total: {len(self.sample_files)} preprocessed {split} samples")

        # Load annotations to get captions
        self.annotations = {}
        for ann in annotations:
            self.annotations[ann["id"]] = ann["label"]
        print(f"Loaded {len(self.annotations)} annotations.")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        """
        Returns:
            features: [seq_len, dim] - Pre-extracted V-JEPA features
            caption: str - Caption text
        """
        pt_file = self.sample_files[idx]
        video_id = pt_file.stem  # Get video ID from filename

        # Load features tensor directly
        features = torch.load(pt_file, map_location="cpu", weights_only=False)
        features = features.squeeze(0)  # Remove batch dimension if present

        caption = random.choice(self.annotations.get(video_id, ""))
        return features, caption
