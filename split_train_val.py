import json, random

INP = "train.jsonl"
TRAIN = "train.jsonl"
VAL = "val.jsonl"
VAL_RATIO = 0.05

with open(INP, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

random.shuffle(rows)
n_val = max(1, int(len(rows) * VAL_RATIO))
val_rows = rows[:n_val]
train_rows = rows[n_val:]

with open(TRAIN, "w", encoding="utf-8") as f:
    for r in train_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(VAL, "w", encoding="utf-8") as f:
    for r in val_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"train={len(train_rows)} val={len(val_rows)}")
