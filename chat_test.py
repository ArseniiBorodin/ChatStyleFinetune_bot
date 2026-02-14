import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_DIR = "out_style_lora_phi3/final"

use_mps = torch.backends.mps.is_available()
device_map = {"": "mps"} if use_mps else {"": "cpu"}
dtype = torch.float16 if use_mps else torch.float32

tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map=device_map,
    torch_dtype=dtype,
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

SYSTEM = "Ты отвечаешь в стиле переписки Arsenii." \
         "Отвечай по смыслу последнего сообщения и контекста." \
         "Не отвечай односложно."

history = [{"role": "system", "content": SYSTEM}]

MAX_MESSAGES = 10  # держим короткую историю

def trim():
    global history
    if len(history) > 1 + MAX_MESSAGES:
        history = [history[0]] + history[-MAX_MESSAGES:]

def ask(user_text: str) -> str:
    global history
    history.append({"role": "user", "content": user_text})
    trim()

    prompt = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    # перенос на устройство
    if use_mps:
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            use_cache=False,
        )

    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    assistant = tok.decode(new_tokens, skip_special_tokens=True).strip()


    history.append({"role": "assistant", "content": assistant })
    trim()
    return assistant

print("Пиши сообщение. Выход: /exit")
while True:
    u = input("\nYou: ").strip()
    if u.lower() in {"/exit", "exit", "quit"}:
        break
    print("\nBot:", ask(u))
