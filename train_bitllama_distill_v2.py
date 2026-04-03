import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    default_data_collator,
    set_seed,
)

from modeling_bitllama_distill import BitLlamaConfig, BitLlamaForCausalLM


# =========================================================
# Args
# =========================================================

@dataclass
class ScriptArgs:
    teacher_model_name: str = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    dataset_name: str = field(default="wikitext")
    dataset_config: str = field(default="wikitext-2-raw-v1")
    output_dir: str = field(default="./bitllama_out_v2")

    max_seq_length: int = field(default=512)
    vocab_size: int = field(default=32000)

    hidden_size: int = field(default=768)
    intermediate_size: int = field(default=2048)
    num_hidden_layers: int = field(default=12)
    num_attention_heads: int = field(default=12)
    group_size: int = field(default=128)

    ce_weight: float = field(default=0.5)
    kl_weight: float = field(default=0.4)
    hidden_weight: float = field(default=0.1)
    distill_temperature: float = field(default=2.0)

    seed: int = field(default=42)
    resume_from_checkpoint: Optional[str] = field(default=None)


# =========================================================
# Tokenization
# =========================================================

def tokenize_fn(examples, tokenizer, max_seq_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )


def add_labels(example):
    example["labels"] = example["input_ids"].copy()
    return example


# =========================================================
# Callback to print extra losses cleanly
# =========================================================

class DistillMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        keys = ["loss", "ce_loss", "kl_loss", "hidden_loss", "eval_loss"]
        filtered = {k: v for k, v in logs.items() if k in keys}
        if filtered:
            print(f"[distill-metrics] step={state.global_step} {filtered}")


# =========================================================
# Trainer
# =========================================================

class DistillTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        labels = inputs["labels"]

        with torch.no_grad():
            teacher_out = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            teacher_logits = teacher_out.logits
            teacher_hidden_states = teacher_out.hidden_states

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            teacher_logits=teacher_logits,
            teacher_hidden_states=teacher_hidden_states,
            output_hidden_states=True,
        )
        loss = outputs["loss"]

        # log decomposed losses when present
        log_dict = {}
        for k in ["ce_loss", "kl_loss", "hidden_loss"]:
            v = outputs.get(k, None)
            if v is not None:
                log_dict[k] = v.detach().float().item()
        if log_dict:
            self.log(log_dict)

        return (loss, outputs) if return_outputs else loss


# =========================================================
# Main
# =========================================================

def main():
    parser = HfArgumentParser((ScriptArgs, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(script_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(script_args.teacher_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        script_args.teacher_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    teacher.eval()

    vocab_size = max(script_args.vocab_size, len(tokenizer))

    config = BitLlamaConfig(
        vocab_size=vocab_size,
        hidden_size=script_args.hidden_size,
        intermediate_size=script_args.intermediate_size,
        num_hidden_layers=script_args.num_hidden_layers,
        num_attention_heads=script_args.num_attention_heads,
        max_position_embeddings=script_args.max_seq_length,
        group_size=script_args.group_size,
        ce_weight=script_args.ce_weight,
        kl_weight=script_args.kl_weight,
        hidden_weight=script_args.hidden_weight,
        distill_temperature=script_args.distill_temperature,
        output_hidden_states=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    student = BitLlamaForCausalLM(config)
    student.tie_weights()

    raw = load_dataset(script_args.dataset_name, script_args.dataset_config)

    tokenized = raw.map(
        lambda x: tokenize_fn(x, tokenizer, script_args.max_seq_length),
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    tokenized = tokenized.map(add_labels)

    trainer = DistillTrainer(
        teacher_model=teacher,
        model=student,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if "validation" in tokenized else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[DistillMetricsCallback()],
    )

    trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
