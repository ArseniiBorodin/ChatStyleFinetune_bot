import json
import re
from typing import List, Dict, Tuple, Optional

# ===== настройки =====
INPUT_FILES = [
    "english/all_chat.txt",
]
OUTPUT = "train_en.jsonl"

SYSTEM = (
    "You're imitating Arsenii's communication style. "
    "Keep the style of the person, but don't lose the context. "
    "Answer in the same language as the user. "
)

ARSENII_NAME = "Arsenii Borodin"   # должно совпадать с именем в txt
MAX_CONTEXT_MESSAGES = 8
MIN_TEXT_CHARS = 1

# формат: [timestamp] Sender: message
pattern = re.compile(r"^\[(.*?)\]\s+(.*?):\s+(.*)$")


def normalize_text(text: str) -> str:
    t = text.replace("\u00a0", " ").strip()
    while "  " in t:
        t = t.replace("  ", " ")
    return t


def parse_txt_files(paths: List[str]) -> List[Tuple[str, str]]:
    msgs: List[Tuple[str, str]] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = pattern.match(line.strip())
                if not m:
                    continue
                sender = m.group(2).strip()
                text = normalize_text(m.group(3))

                if not text or len(text) < MIN_TEXT_CHARS:
                    continue

                # медиа → маркер
                if "image omitted" in text.lower():
                    text = "<IMAGE>"

                msgs.append((sender, text))
    return msgs


def build_block_samples(messages: List[Tuple[str, str]]) -> List[Dict]:
    """
    Делает samples на блоки ответа Arsenii:
      system + последние MAX_CONTEXT_MESSAGES сообщений + (assistant: слитый блок)
    """
    samples: List[Dict] = []
    context: List[Dict[str, str]] = []
    assistant_buf: List[str] = []

    def flush_assistant():
        nonlocal assistant_buf, context, samples
        if not assistant_buf:
            return
        merged = "\n".join(assistant_buf).strip()
        assistant_buf = []
        if not merged:
            return

        # sample только если в контексте есть user
        if any(x["role"] == "user" for x in context):
            sample_msgs = [{"role": "system", "content": SYSTEM}]
            sample_msgs.extend(context[-MAX_CONTEXT_MESSAGES:])
            sample_msgs.append({"role": "assistant", "content": merged})
            samples.append({"messages": sample_msgs})

        # кладем слитый ответ в контекст одной репликой
        context.append({"role": "assistant", "content": merged})

        # ограничим контекст
        if len(context) > MAX_CONTEXT_MESSAGES * 3:
            context[:] = context[-MAX_CONTEXT_MESSAGES:]

    for sender, text in messages:
        role = "assistant" if sender == ARSENII_NAME else "user"

        if role == "assistant":
            assistant_buf.append(text)
        else:
            # user пришел -> закрываем предыдущий блок ответа Arsenii
            flush_assistant()

            context.append({"role": "user", "content": text})
            if len(context) > MAX_CONTEXT_MESSAGES * 3:
                context[:] = context[-MAX_CONTEXT_MESSAGES:]

    flush_assistant()
    return samples


def main():
    msgs = parse_txt_files(INPUT_FILES)
    samples = build_block_samples(msgs)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Done. Wrote {len(samples)} block-samples to {OUTPUT}")


if __name__ == "__main__":
    main()
