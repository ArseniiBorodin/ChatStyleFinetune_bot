import json, re

INP="train.jsonl"
OUT="train_sanitized.jsonl"

re_url = re.compile(r"https?://\S+")
re_iban = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
re_phone = re.compile(r"\+\d{6,15}")

def clean(s):
    s = re_url.sub("<URL>", s)
    s = re_iban.sub("<IBAN>", s)
    s = re_phone.sub("<PHONE>", s)
    return s

with open(INP, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as w:
    for line in f:
        obj = json.loads(line)
        for m in obj["messages"]:
            m["content"] = clean(m["content"])
        w.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("ok ->", OUT)
