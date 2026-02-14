import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

#REDACTED_HF_TOKEN
# =========================
# 1) Настройки
# =========================
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"   # ✅ меньше, чем Mistral 7B
TRAIN_PATH = "train.jsonl"
VAL_PATH   = "val.jsonl"
OUT_DIR    = "out_style_lora_phi3"

MAX_SEQ_LEN = 1024     # можно 512 если будет тяжело
EPOCHS = 2
LR = 2e-4


# =========================
# 2) Устройство (Mac MPS / CPU)
# =========================
use_mps = torch.backends.mps.is_available()
device_map = {"": "mps"} if use_mps else {"": "cpu"}
dtype = torch.float16 if use_mps else torch.float32
print("✅ Using MPS (Apple Silicon GPU)" if use_mps else "⚠️ Using CPU")


# =========================
# 3) Токенайзер и модель
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map=device_map,
    torch_dtype=dtype,
)

model.config.use_cache = False
try:
    model.gradient_checkpointing_enable()
    print("✅ Gradient checkpointing enabled")
except Exception as e:
    print("⚠️ Could not enable gradient checkpointing:", e)


# =========================
# 4) Авто-подбор target_modules для LoRA
#    (чтобы работало на разных архитектурах)
# =========================
def pick_lora_targets(m):
    import torch.nn as nn

    # Смотрим все Linear-слои, берём их "короткие" имена (последний сегмент)
    short_names = set()
    for name, module in m.named_modules():
        if isinstance(module, nn.Linear):
            short_names.add(name.split(".")[-1])

    # Предпочтительные имена (часто встречаются)
    preferred = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "qkv_proj", "Wqkv", "wo", "wq", "wk", "wv",
    ]

    targets = [x for x in preferred if x in short_names]

    # Если ничего не нашли — берём все линейные (но это может быть тяжелее)
    if not targets:
        targets = sorted(short_names)

    print("✅ LoRA target modules:", targets)
    return targets


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=pick_lora_targets(model),
)


# =========================
# 5) Датасет
# =========================
ds = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})

def formatting_func(example):
    msgs = example["messages"]
    # Для SFT: шаблон + уже готовый assistant-ответ внутри сообщений
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


# =========================
# 6) Конфиг обучения
# =========================
cfg = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LR,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    logging_steps=10,
    save_steps=200,
    save_total_limit=2,

    fp16=use_mps,
    bf16=False,
    report_to=[],
)


trainer = SFTTrainer(
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    peft_config=peft_config,
    args=cfg,
    formatting_func=formatting_func,
)

trainer.train()

final_dir = os.path.join(OUT_DIR, "final")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"✅ Done. Saved to: {final_dir}")
