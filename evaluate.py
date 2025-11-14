"""
Video Captioning Evaluation Script

This script evaluates generated video captions using standard NLP metrics designed for
captioning tasks. These metrics measure how well generated captions match ground truth
references, accounting for paraphrasing and semantic similarity.

Metrics and Benchmarks:
-----------------------

1. BLEU (Bilingual Evaluation Understudy)
   - BLEU-1/2/3/4: Measures n-gram precision (1-gram, 2-gram, 3-gram, 4-gram overlap)
   - Range: 0-1 (higher is better)
   - Decent score: BLEU-4 > 0.30 for video captioning
   - Note: Strict metric, penalizes paraphrasing heavily

2. METEOR (Metric for Evaluation of Translation with Explicit ORdering)
   - Considers synonyms, stemming, and paraphrases
   - Range: 0-1 (higher is better)
   - Decent score: > 0.25 for video captioning
   - Better than BLEU for capturing semantic similarity

3. ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)
   - Measures longest common subsequence between generated and reference
   - Range: 0-1 (higher is better)
   - Decent score: > 0.50 for video captioning
   - Good for measuring word order and fluency

4. CIDEr (Consensus-based Image Description Evaluation)
   - THE GOLD STANDARD for image/video captioning
   - Measures consensus with human-written captions using TF-IDF weighting
   - Range: 0-10 (higher is better, can exceed 10 in some datasets)
   - Decent score: > 0.80 for video captioning (dataset-dependent)
   - Excellent score: > 1.20
   - Note: Most captioning papers report CIDEr as the primary metric

Typical benchmarks for video captioning tasks:
- Strong model: BLEU-4: 0.40+, METEOR: 0.30+, CIDEr: 1.00+
- Decent model: BLEU-4: 0.25-0.35, METEOR: 0.22-0.28, CIDEr: 0.60-0.90
- Baseline: BLEU-4: 0.15-0.25, METEOR: 0.15-0.22, CIDEr: 0.30-0.60

Usage:
    pip install pycocoevalcap
    python evaluate.py
"""

import torch
from pathlib import Path
from dataset import CaptionedClipsPreprocessed
from models.model import VideoCaptionModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def evaluate(model, dataloader, device):
    """
    Evaluate model on validation set using standard captioning metrics.

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    model.eval()

    gts = {}  # ground truth captions
    res = {}  # generated captions

    print("Generating captions for validation set...")
    with torch.no_grad():
        for idx, (features_batch, captions) in enumerate(tqdm(dataloader)):
            features_batch = features_batch.to(device)

            # Generate captions
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                generated = model.generate(features_batch, max_new_tokens=50)

            # Store results in format expected by pycocoevalcap
            for i, (gt, pred) in enumerate(zip(captions, generated)):
                sample_id = idx * len(captions) + i
                gts[sample_id] = [gt]  # List of reference captions (can have multiple)
                res[sample_id] = [pred]  # List with single generated caption

    print(f"\nEvaluating {len(gts)} samples...")
    print("Computing metrics (this may take a minute)...")

    # Compute metrics
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"], "BLEU"),
        (Meteor(), "METEOR", "METEOR"),
        (Rouge(), "ROUGE_L", "ROUGE-L"),
        (Cider(), "CIDEr", "CIDEr"),
    ]

    scores = {}
    for scorer, method, display_name in scorers:
        print(f"  Computing {display_name}...", end=" ", flush=True)
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            # BLEU returns multiple scores
            for m, s in zip(method, score):
                scores[m] = s
        else:
            scores[method] = score
        print("✓")

    return scores, gts, res


if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8

    # Setup validation dataset
    preprocessed_root_1 = "/media/steve/storage2/vjepa_features/"
    preprocessed_root_2 = "/media/steve/storage/vjepa_features/"
    annotations_path = "/media/steve/storage/20bn-something-something-v2-labels"

    annotations_val = json.loads(
        (Path(annotations_path) / "validation.json").read_text()
    )

    for ann in annotations_val:
        del ann["template"]
        ann["label"] = [ann["label"]]

    val_dataset = CaptionedClipsPreprocessed(
        preprocessed_root_1,
        split="val",
        preprocessed_root_path_2=preprocessed_root_2,
        annotations=annotations_val,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    # Load model from checkpoint
    print("Loading model from checkpoint...")
    model = VideoCaptionModel(device=device, load_vision=False)

    checkpoint_path = Path("checkpoint_supp.pt.bak")
    if not checkpoint_path.exists():
        raise FileNotFoundError("No checkpoint found at checkpoint.pt")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Run evaluation
    scores, gts, res = evaluate(model, val_dataloader, device)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in scores.items():
        print(f"{metric:12s}: {score:.4f}")
    print("=" * 50)

    # Save detailed results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / f"evaluation_epoch_{checkpoint['epoch']}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "epoch": checkpoint["epoch"],
                "metrics": scores,
                "sample_predictions": {
                    k: {"ground_truth": gts[k][0], "prediction": res[k][0]}
                    for k in list(gts.keys())[:20]  # Save first 20 samples
                },
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to {output_file}")

    # Print some sample predictions
    print("\nSample Predictions:")
    print("-" * 50)
    for i in range(min(5, len(gts))):
        print(f"\nSample {i + 1}:")
        print(f"  GT:   {gts[i][0]}")
        print(f"  Pred: {res[i][0]}")
