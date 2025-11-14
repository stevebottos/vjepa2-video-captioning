"""
Inference on Something-Something-V2 test set samples.

This script:
1. Loads k random videos from SSv2 test set
2. Processes each video as a single segment (same as training)
3. Generates captions
4. Saves both the video IDs and captions to results/

Usage:
    python infer_ssv2.py --num-samples 10
    python infer_ssv2.py --num-samples 50 --checkpoint checkpoint_epoch_10.pt
"""

import torch
import argparse
import json
import random
import csv
import shutil
from pathlib import Path
from torchcodec.decoders import VideoDecoder
import numpy as np
from transformers import AutoVideoProcessor, AutoModel
from models.model import VideoCaptionModel
from torchvision.transforms import v2 as transforms
from tqdm import tqdm


def extract_frames(video_path, num_frames=32, target_size=256):
    """
    Extract uniformly sampled frames from video (same as training preprocessing).

    Args:
        video_path: Path to video file
        num_frames: Number of frames to uniformly sample
        target_size: Target size to resize frames to

    Returns:
        torch.Tensor: Frames tensor [num_frames, C, H, W]
    """
    decoder = VideoDecoder(video_path)
    total_frames = len(decoder)

    # Uniformly sample frames
    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    # Extract frames
    frames = decoder.get_frames_at(frame_indices).data  # [N, C, H, W]

    # Resize frames
    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size), antialias=True),
        ]
    )
    frames = transform(frames)

    # Pad if necessary
    if len(frames) < num_frames:
        num_padding = num_frames - len(frames)
        black_frame = torch.zeros_like(frames[0])
        padding = black_frame.unsqueeze(0).repeat(num_padding, 1, 1, 1)
        frames = torch.cat([frames, padding], dim=0)

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=60)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--test-json", type=str, default="/media/steve/storage/20bn-something-something-v2-labels/test.json")
    parser.add_argument("--test-answers", type=str, default="/media/steve/storage/20bn-something-something-v2-labels/test-answers.csv")
    parser.add_argument("--videos-path", type=str, default="/media/steve/storage/20bn-something-something-v2")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading test annotations from {args.test_json}")
    test_annotations = json.loads(Path(args.test_json).read_text())
    print(f"Total test videos: {len(test_annotations)}")

    print(f"Loading ground truth labels from {args.test_answers}")
    ground_truth_labels = {}
    with open(args.test_answers, "r") as f:
        csv_reader = csv.reader(f, delimiter=";")
        next(csv_reader)
        for row in csv_reader:
            if len(row) >= 2:
                video_id = row[0]
                label = row[1]
                ground_truth_labels[video_id] = label
    print(f"Loaded {len(ground_truth_labels)} ground truth labels")

    videos_path = Path(args.videos_path)
    existing_annotations = []
    for ann in test_annotations:
        video_path = videos_path / f"{ann['id']}.webm"
        if video_path.exists():
            existing_annotations.append(ann)

    print(f"Videos found on disk: {len(existing_annotations)}")

    if args.num_samples > len(existing_annotations):
        print(
            f"Warning: Requested {args.num_samples} samples but only {len(existing_annotations)} available"
        )
        args.num_samples = len(existing_annotations)

    sampled_annotations = random.sample(existing_annotations, args.num_samples)
    print(f"Sampled {len(sampled_annotations)} random videos\n")

    print("Loading V-JEPA model...")
    vjepa_processor = AutoVideoProcessor.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256"
    )
    vjepa_model = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        dtype=torch.float16,
        device_map=str(device),
        attn_implementation="sdpa",
    )
    vjepa_model.eval()

    print("Loading caption model...")
    model = VideoCaptionModel(device=device, load_vision=False)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}\n")

    results = []
    print("Generating captions...")

    for ann in tqdm(sampled_annotations):
        video_id = ann["id"]
        ground_truth = ground_truth_labels.get(video_id, "[NO GROUND TRUTH]")
        video_path = videos_path / f"{video_id}.webm"

        try:
            frames = extract_frames(str(video_path), num_frames=args.num_frames)

            with torch.no_grad():
                processed = vjepa_processor(frames, return_tensors="pt")
                processed = {k: v.to(device) for k, v in processed.items()}

                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    features = vjepa_model.get_vision_features(**processed)

                features = features.squeeze(0).unsqueeze(0)

                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    prediction = model.generate(features, max_new_tokens=50)[0]

            results.append(
                {
                    "video_id": video_id,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                }
            )

        except Exception as e:
            print(f"\nError processing {video_id}: {e}")
            results.append(
                {
                    "video_id": video_id,
                    "ground_truth": ground_truth,
                    "prediction": f"[ERROR: {str(e)}]",
                }
            )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    run_dir = results_dir / f"ssv2_test_inference_{args.num_samples}_samples_epoch_{checkpoint['epoch']}"
    run_dir.mkdir(exist_ok=True)

    videos_dir = run_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    print(f"\nCopying video files to {videos_dir}...")
    for result in tqdm(results, desc="Copying videos"):
        video_id = result["video_id"]
        src_video = videos_path / f"{video_id}.webm"
        dst_video = videos_dir / f"{video_id}.webm"
        if src_video.exists():
            shutil.copy2(src_video, dst_video)

    output_file = run_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "checkpoint_epoch": checkpoint["epoch"],
                "num_samples": args.num_samples,
                "seed": args.seed,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print("=" * 70)
    print(f"Total samples processed: {len(results)}")
    print(f"Results saved to: {run_dir}")
    print(f"  - JSON: {output_file}")
    print(f"  - Videos: {videos_dir}\n")

    print("Sample predictions:")
    print("-" * 70)
    for i, result in enumerate(results[:5]):
        print(f"\nSample {i + 1} (ID: {result['video_id']}):")
        print(f"  GT:   {result['ground_truth']}")
        print(f"  Pred: {result['prediction']}")

    if len(results) > 5:
        print(f"\n... and {len(results) - 5} more (see {output_file})")
    print("=" * 70)


if __name__ == "__main__":
    main()
