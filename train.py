import os
import torch
import json
import argparse
from pathlib import Path
from dataset import CaptionedClipsPreprocessed
from models.model import VideoCaptionModel
from torch.utils.data import DataLoader
from tqdm import tqdm


def memops():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)


def validate(model, val_dataloader, device):
    """
    Run validation loop on entire validation set.

    Args:
        model: The model to validate
        val_dataloader: Validation dataloader
        device: Device to run on

    Returns:
        tuple: (avg_val_loss, all_captions)
    """
    model.eval()
    total_val_loss = 0
    num_batches = 0
    all_captions = []

    with torch.no_grad():
        val_progress = tqdm(
            val_dataloader,
            desc="Validating",
            total=len(val_dataloader),
        )

        for features_batch, captions in val_progress:
            # Features are already pre-extracted, just move to device
            features_batch = features_batch.to(device)

            # Compute validation loss
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = model(features_batch, captions)

            total_val_loss += loss.item()
            num_batches += 1

            # Generate captions for all validation samples
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                generated = model.generate(features_batch, max_new_tokens=50)
            for gt, pred in zip(captions, generated):
                all_captions.append({"ground_truth": gt, "prediction": pred})

            val_progress.set_postfix({"val_loss": f"{loss.item():.4f}"})

    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    model.train()

    return avg_val_loss, all_captions


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train video captioning model")
    parser.add_argument(
        "--eval-only", action="store_true", help="Skip training and only run evaluation"
    )
    args = parser.parse_args()

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    gradient_accumulation_steps = 16  # Accumulate gradients over 16 batches
    learning_rate = 5e-5  # Lower LR to reduce overfitting
    num_epochs = 100

    # Setup dataset and dataloader with preprocessed features from both disks
    preprocessed_root_1 = "/media/steve/storage2/vjepa_features/"
    preprocessed_root_2 = "/media/steve/storage/vjepa_features/"
    annotations_path = "/media/steve/storage/20bn-something-something-v2-labels"

    annotations_train = json.loads((Path(annotations_path) / "train.json").read_text())
    train_paraphrases = json.loads(Path("ssv2_paraphrase.json").read_text())

    for ann in annotations_train:
        del ann["template"]
        if ann["id"] in train_paraphrases:
            ann["label"] = [ann["label"]] + train_paraphrases[ann["id"]]
        else:
            ann["label"] = [ann["label"]]

    dataset = CaptionedClipsPreprocessed(
        preprocessed_root_1,
        split="train",
        preprocessed_root_path_2=preprocessed_root_2,
        annotations=annotations_train,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

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

    # Initialize model and optimizer (load_vision=False to skip loading V-JEPA)
    memops()
    model = VideoCaptionModel(device=device, load_vision=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.03
    )

    # Resume from checkpoint if available
    start_epoch = 0
    if os.path.exists("checkpoint.pt"):
        checkpoint = torch.load("checkpoint.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        print(f"Loaded from epoch {start_epoch - 1}")
        print(f"Previous train loss: {checkpoint['train_loss']:.4f}")
        if "val_loss" in checkpoint:
            print(f"Previous val loss: {checkpoint['val_loss']:.4f}\n")
    else:
        print("\nNo checkpoint found. Starting training from scratch.\n")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        if not args.eval_only:
            model.train()
            total_loss = 0
            accumulated_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for step, (features_batch, captions) in enumerate(progress_bar):
                # Features are already pre-extracted, just move to device
                features_batch = features_batch.to(device)
                # Forward pass with automatic mixed precision
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    loss = model(features_batch, captions)

                # Scale loss for gradient accumulation
                scaled_loss = loss / gradient_accumulation_steps

                # Backward pass (accumulate gradients)
                scaled_loss.backward()

                # Accumulate loss for logging
                accumulated_loss += loss.item()

                # Optimizer step every gradient_accumulation_steps
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    # Update metrics with accumulated loss
                    avg_accumulated_loss = (
                        accumulated_loss / gradient_accumulation_steps
                    )
                    total_loss += accumulated_loss
                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_accumulated_loss:.4f}",
                            "step": f"{(step + 1) // gradient_accumulation_steps}",
                        }
                    )
                    accumulated_loss = 0
            # Handle remaining gradients at end of epoch
            if (step + 1) % gradient_accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
                total_loss += accumulated_loss
            avg_loss = total_loss / len(dataloader)
            print(f"\nEpoch {epoch + 1} - Train Loss: {avg_loss:.4f}")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_loss,
                },
                f"checkpoint.pt",
            )

        # Validation
        print("Running validation...")
        val_loss, sample_captions = validate(model, val_dataloader, device)
        print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")

        # Log losses to simple text file
        with open("losses.txt", "a") as f:
            f.write(
                f"Epoch {epoch + 1}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}\n"
            )

        # Save all validation predictions to file
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        results_file = results_dir / f"validation_epoch_{epoch + 1}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "epoch": epoch + 1,
                    "predictions": sample_captions,
                },
                f,
                indent=2,
            )
        print(f"\nValidation results saved to {results_file}")

        # Print sample generations
        print("\nSample Generations:")
        for i, sample in enumerate(sample_captions[:3]):  # Show first 3
            print(f"\n  Sample {i + 1}:")
            print(f"    GT:   {sample['ground_truth']}")
            print(f"    Pred: {sample['prediction']}")

        if args.eval_only:
            break
    print("Training complete!")
