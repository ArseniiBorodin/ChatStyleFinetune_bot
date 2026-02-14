import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


SYSTEM = (
    "Ты отвечаешь в стиле переписки Arsenii. "
    "Отвечай по смыслу и используй контекст. "
    "Отвечай на том же языке, на котором тебе пишут."
)

# Arsenii = assistant, все остальные = user (могут меняться)
ARSENII_FROM_IDS = {"user750091526"}  # добавь сюда другие from_id Arsenii, если есть
ARSENII_FROM_NAMES = {"Arsenii Börodin", "Arsenii Borodin", "Arsenii Börodin"}

# Контекст
MAX_CONTEXT_MESSAGES = 8   # сколько последних сообщений (без system) давать в пример
MIN_TEXT_CHARS = 1


def extract_text(text_field: Any) -> Optional[str]:
    if text_field is None:
        return None
    if isinstance(text_field, str):
        s = text_field.strip()
        return s if s else None
    if isinstance(text_field, list):
        parts: List[str] = []
        for x in text_field:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict) and "text" in x:
                parts.append(str(x["text"]))
        s = "".join(parts).strip()
        return s if s else None
    return None


def is_arsenii(msg: Dict[str, Any]) -> bool:
    fid = msg.get("from_id")
    frm = msg.get("from")
    if fid and fid in ARSENII_FROM_IDS:
        return True
    if frm and frm in ARSENII_FROM_NAMES:
        return True
    return False


def normalize_text(text: str) -> str:
    t = text.replace("\u00a0", " ").strip()
    while "  " in t:
        t = t.replace("  ", " ")
    return t


def load_one_or_many_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    # 1) обычный JSON (объект или массив)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass

    # 2) NDJSON: по одному JSON-объекту на строку
    out: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def build_context_samples_from_chat(chat_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    msgs = chat_obj.get("messages", [])
    samples: List[Dict[str, Any]] = []

    # контекст — последние сообщения (user/assistant), но без system
    context: List[Dict[str, str]] = []

    # буфер для склейки ответов Arsenii
    assistant_buf: List[str] = []

    def flush_assistant_buf():
        """Закрываем блок ответа Arsenii: создаём один sample и кладём слитый ответ в контекст."""
        nonlocal assistant_buf, context, samples
        if not assistant_buf:
            return

        merged = "\n".join(assistant_buf).strip()
        assistant_buf = []
        if not merged:
            return

        # создаём sample только если в контексте есть user (иначе нечему отвечать)
        if any(x["role"] == "user" for x in context):
            sample_msgs = [{"role": "system", "content": SYSTEM}]
            sample_msgs.extend(context[-MAX_CONTEXT_MESSAGES:])
            sample_msgs.append({"role": "assistant", "content": merged})
            samples.append({"messages": sample_msgs})

        # добавляем слитый ответ в контекст одной репликой
        context.append({"role": "assistant", "content": merged})

        # ограничиваем контекст
        if len(context) > MAX_CONTEXT_MESSAGES * 3:
            context[:] = context[-MAX_CONTEXT_MESSAGES:]

    last_role: Optional[str] = None

    for m in msgs:
        if m.get("type") != "message":
            continue

        text = extract_text(m.get("text"))
        if text is None:
            continue
        text = normalize_text(text)
        if len(text) < MIN_TEXT_CHARS:
            continue

        role = "assistant" if is_arsenii(m) else "user"

        if role == "assistant":
            # если ассистент продолжает писать — копим в буфер
            assistant_buf.append(text)
            last_role = "assistant"
        else:
            # пришёл user -> сначала закрываем предыдущий ответ Arsenii (если он был)
            flush_assistant_buf()

            # кладём user-сообщение в контекст
            context.append({"role": "user", "content": text})
            last_role = "user"

            if len(context) > MAX_CONTEXT_MESSAGES * 3:
                context[:] = context[-MAX_CONTEXT_MESSAGES:]

    # закрываем буфер в конце
    flush_assistant_buf()

    return samples


def convert(input_paths: List[str], output_path: str = "train.jsonl") -> int:
    all_samples: List[Dict[str, Any]] = []

    for ip in input_paths:
        chat_objs = load_one_or_many_json(ip)
        for chat in chat_objs:
            all_samples.extend(build_context_samples_from_chat(chat))

    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Done. Wrote {len(all_samples)} context samples to {output_path}")
    return len(all_samples)


if __name__ == "__main__":
    convert(
        [
            "russian/result.json",
            "russian/result2.json",
            "russian/result3.json",
            "russian/result4.json",
            "russian/result5.json",
            "russian/result6.json",
            "russian/result7.json",
        ],
        "../train_ru.jsonl"
    )
