import torch
from torch import nn
from transformers import (
    AutoVideoProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from .qformer import QFormer


class VideoCaptionModel(nn.Module):
    def __init__(self, device, model_name="Qwen/Qwen2-1.5B", load_vision=False):
        super().__init__()

        self.device = device
        self.load_vision = load_vision

        if load_vision:
            # Vision encoder (V-JEPA) - load for preprocessing
            self.raw_input_processor = AutoVideoProcessor.from_pretrained(
                "facebook/vjepa2-vitl-fpc64-256"
            )
            self.video_tokenizer = AutoModel.from_pretrained(
                "facebook/vjepa2-vitl-fpc64-256",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).to(device)

            # Freeze vision encoder
            for param in self.video_tokenizer.parameters():
                param.requires_grad = False

        # Decoder-only language model
        self.text_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)

        # Freeze language model (only train Q-Former)
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if not present
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        lm_hidden_dim = self.text_model.config.hidden_size
        self.qformer = QFormer(
            num_queries=64,
            dim=lm_hidden_dim,
            visual_dim=1024,
            num_heads=4,
            depth=1,
        ).to(device)

        # Tokenize prompt prefix: "a video showing:"
        # These tokens will be prepended before vision embeddings
        # ... We're doing this so that there's some "known" language component,
        # empirically this works better
        prompt_text = "a video showing:"
        self.prompt_tokens = self.text_tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)
        print(
            f"  Prompt prefix: '{prompt_text}' ({self.prompt_tokens.shape[1]} tokens)"
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable (Q-Former only): {trainable_params:,}")
        print(f"  Frozen: {total_params - trainable_params:,}")
        print(f"  LM hidden dim: {lm_hidden_dim}")

    @torch.no_grad()
    def get_vision_features(self, raw_input_frames):
        video = self.raw_input_processor(raw_input_frames, return_tensors="pt").to(
            self.device
        )
        features = self.video_tokenizer.get_vision_features(**video)
        return features

    def forward(self, vision_features, captions):
        batch_size = vision_features.shape[0]

        # Get prompt embeddings: "a video showing:"
        prompt_embeds = self.text_model.get_input_embeddings()(
            self.prompt_tokens.expand(batch_size, -1)
        )
        num_prompt_tokens = prompt_embeds.shape[1]

        # Get vision embeddings from Q-Former
        vision_embeds = self.qformer(vision_features.to(self.device))
        num_vision_tokens = vision_embeds.shape[1]

        # Tokenize captions - manually append EOS token
        # Qwen2 tokenizer doesn't add EOS with add_special_tokens=True
        captions_with_eos = [
            c.strip() + self.text_tokenizer.eos_token for c in captions
        ]
        caption_tokens = self.text_tokenizer(
            captions_with_eos,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        ).to(self.device)

        # Get text embeddings from the LM's embedding layer
        text_embeds = self.text_model.get_input_embeddings()(caption_tokens.input_ids)

        # Concatenate: vision + prompt + text embeddings
        inputs_embeds = torch.cat([vision_embeds, prompt_embeds, text_embeds], dim=1)

        # Create labels: -100 for vision and prompt tokens (ignored), actual token IDs for text
        vision_labels = torch.full(
            (batch_size, num_vision_tokens), -100, dtype=torch.long, device=self.device
        )
        prompt_labels = torch.full(
            (batch_size, num_prompt_tokens), -100, dtype=torch.long, device=self.device
        )
        labels = torch.cat(
            [vision_labels, prompt_labels, caption_tokens.input_ids], dim=1
        )  # [batch, num_vision + num_prompt + text_len]

        # Create attention mask: 1 for all tokens (vision + prompt + text)
        vision_attention_mask = torch.ones(
            (batch_size, num_vision_tokens), dtype=torch.long, device=self.device
        )
        prompt_attention_mask = torch.ones(
            (batch_size, num_prompt_tokens), dtype=torch.long, device=self.device
        )
        attention_mask = torch.cat(
            [
                vision_attention_mask,
                prompt_attention_mask,
                caption_tokens.attention_mask,
            ],
            dim=1,
        )

        output = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return output.loss

    def generate(self, vision_features, max_new_tokens=50, num_beams=1):
        """
        Generate captions autoregressively.

        Args:
            vision_features: [batch, num_vision_tokens, 1024] - Vision features from video encoder
            max_new_tokens: Maximum number of new tokens to generate
            num_beams: Number of beams for beam search (1 = greedy)

        Returns:
            captions: List of generated caption strings
        """
        batch_size = vision_features.shape[0]

        # Get prompt embeddings: "a video showing:"
        prompt_embeds = self.text_model.get_input_embeddings()(
            self.prompt_tokens.expand(batch_size, -1)
        )  # [batch, num_prompt_tokens, hidden]
        num_prompt_tokens = prompt_embeds.shape[1]

        # Compress and project vision features using Q-Former
        vision_embeds = self.qformer(vision_features.to(self.device))
        num_vision_tokens = vision_embeds.shape[1]

        # Concatenate vision + prompt embeddings
        inputs_embeds = torch.cat([vision_embeds, prompt_embeds], dim=1)

        # Create attention mask for vision + prompt tokens
        vision_attention_mask = torch.ones(
            (batch_size, num_vision_tokens), dtype=torch.long, device=self.device
        )
        prompt_attention_mask = torch.ones(
            (batch_size, num_prompt_tokens), dtype=torch.long, device=self.device
        )
        attention_mask = torch.cat(
            [vision_attention_mask, prompt_attention_mask], dim=1
        )

        # Generate from prompt + vision embeddings
        output_ids = self.text_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            pad_token_id=self.text_tokenizer.pad_token_id,
            eos_token_id=self.text_tokenizer.eos_token_id,
            early_stopping=True,
        )

        captions = self.text_tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        return captions
