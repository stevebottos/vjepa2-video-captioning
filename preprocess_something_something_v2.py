from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from torchvision.transforms import v2 as transforms

from torchcodec.decoders import VideoDecoder

annotations_path = Path("/media/steve/storage/20bn-something-something-v2-labels")
webms_path = Path("/media/steve/storage/20bn-something-something-v2")
output_root_1 = Path("/media/steve/storage2/vjepa_features")
output_root_2 = Path("/media/steve/storage/vjepa_features")


class SSV2FrameDataset(Dataset):
    """
    Something-Something-V2 dataset that extracts frames only (no preprocessing).
    Use with DataLoader for parallel frame extraction.
    """

    def __init__(
        self,
        annotations_path,
        videos_path,
        split="train",
        num_frames=32,
        target_size=256,
        already_processed=None,
    ):
        """
        Args:
            annotations_path: Path to annotations directory
            videos_path: Path to videos directory
            split: "train" or "validation"
            num_frames: Number of frames to uniformly sample
            target_size: Target size to resize frames to (width and height)
            already_processed: Set of video IDs to skip
        """
        self.videos_path = Path(videos_path)
        self.num_frames = num_frames
        self.split = split
        self.transform = transforms.Compose(
            [
                transforms.Resize((target_size, target_size), antialias=True),
            ]
        )

        # Load annotations
        annotations_file = Path(annotations_path) / f"{split}.json"
        annotations = json.loads(annotations_file.read_text())

        # Filter to only videos that exist and haven't been processed
        already_processed = already_processed or set()
        self.samples = []
        for ann in annotations:
            video_id = ann["id"]
            if video_id in already_processed:
                continue
            video_path = self.videos_path / f"{video_id}.webm"
            if video_path.exists():
                self.samples.append(
                    {
                        "video_id": video_id,
                        "video_path": str(video_path),
                        "caption": ann["label"],
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        video_id = sample["video_id"]
        caption = sample["caption"]

        # Extract frames using torchcodec
        try:
            decoder = VideoDecoder(video_path)
            total_frames = len(decoder)

            if total_frames < self.num_frames:
                # Sample all frames if video is too short
                frame_indices = list(range(total_frames))
            else:
                # Uniformly sample frames
                frame_indices = np.linspace(
                    0, total_frames - 1, self.num_frames, dtype=int
                ).tolist()

            # Extract frames as tensors [N, C, H, W]
            frames = decoder.get_frames_at(frame_indices).data

            # Resize frames using torchvision transforms
            frames = self.transform(frames)  # [N, C, 256, 256]

            # Handle padding if needed
            if len(frames) < self.num_frames:
                num_padding = self.num_frames - len(frames)
                # Create black padding frames [num_padding, C, 256, 256]
                black_frame = torch.zeros_like(frames[0])
                padding = black_frame.unsqueeze(0).repeat(num_padding, 1, 1, 1)
                frames = torch.cat([frames, padding], dim=0)

                # Create mask: False for real frames, True for padded (to be ignored)
                mask = torch.tensor(
                    [False] * (self.num_frames - num_padding) + [True] * num_padding
                )
            else:
                # No padding needed - all frames are real (none should be masked)
                mask = torch.tensor([False] * self.num_frames)

            return {
                "frames": frames,  # [num_frames, C, 256, 256]
                "mask": mask,
                "caption": caption,
                "video_id": video_id,
            }

        except Exception as e:
            # Let the error propagate so we can see what's wrong
            print(f"Error loading {video_id}: {e}")
            raise


if __name__ == "__main__":
    from tqdm import tqdm
    from transformers import AutoVideoProcessor, AutoModel

    # Load V-JEPA model
    processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    model = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()

    for subset in ["train", "validation"]:
        output_1 = output_root_1 / f"ssv2_{subset}"
        output_2 = output_root_2 / f"ssv2_{subset}"
        output_1.mkdir(parents=True, exist_ok=True)
        output_2.mkdir(parents=True, exist_ok=True)

        # Check for already processed files in both directories
        already_processed = set()
        for f in output_1.glob("*.pt"):
            already_processed.add(f.stem)
        for f in output_2.glob("*.pt"):
            already_processed.add(f.stem)

        # Count total processed in output_1 to know when to switch to output_2
        processed_in_output_1 = len(list(output_1.glob("*.pt")))

        print(f"Already processed {len(already_processed)} videos in {subset}")
        print(
            f"  - {processed_in_output_1} in output_1, {len(already_processed) - processed_in_output_1} in output_2"
        )

        # Create dataset and dataloader with multiple workers for parallel frame extraction
        dataset = SSV2FrameDataset(
            annotations_path=annotations_path,
            videos_path=webms_path,
            split=subset,
            num_frames=32,
            already_processed=already_processed,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=8,  # Parallel frame extraction
            shuffle=False,
            pin_memory=True,
            prefetch_factor=1,
        )

        print(f"Processing {len(dataset)} videos in {subset}")

        # Track how many we've saved to know which disk to use
        saved_count = processed_in_output_1

        for batch in tqdm(dataloader, desc=f"Processing {subset}"):
            frames_batch = batch["frames"]  # [B, num_frames, C, 256, 256]
            masks_batch = batch["mask"]  # [B, num_frames]
            captions_batch = batch["caption"]
            video_ids_batch = batch["video_id"]

            # Process each video in the batch through V-JEPA
            with torch.no_grad():
                for i in range(len(frames_batch)):
                    frames = frames_batch[i]  # [num_frames, C, 256, 256]
                    mask = masks_batch[i]  # [num_frames]
                    caption = captions_batch[i]
                    video_id = video_ids_batch[i]

                    # Process with V-JEPA
                    proc_clip = processor(frames, return_tensors="pt")
                    proc_clip = {k: v.to(model.device) for k, v in proc_clip.items()}
                    features = model.get_vision_features(**proc_clip).cpu()

                    # Choose output directory: first 100k to output_1, rest to output_2
                    # output = output_1 if saved_count < 100000 else output_2
                    output = output_2
                    output_file = output / f"{video_id}.pt"
                    torch.save(features, output_file)

                    saved_count += 1

            torch.cuda.empty_cache()
