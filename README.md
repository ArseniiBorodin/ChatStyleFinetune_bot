# ChatStyleFinetune_bot — LoRA Style Chatbot

A local LLM fine-tuning pipeline for imitating a personal messaging style using LoRA.

This project provides tools to:

1. Convert exported chat data (Telegram / WhatsApp) into training format
2. Sanitize and clean conversation data
3. Split dataset into training and validation sets
4. Fine-tune a language model using LoRA
5. Run and test the trained model locally

Designed to run efficiently on macOS (Apple Silicon with MPS), but also works on CPU.

---

## 🧠 How It Works

The system follows a structured pipeline:

1. **Raw chat export**
   Export conversations from Telegram or WhatsApp.

2. **Conversion**
   Convert chat data into a structured `messages` format compatible with chat-based LLM fine-tuning.

3. **Sanitization**
   Remove unwanted content such as:

   * Empty messages
   * Media placeholders
   * Sensitive information
   * Extremely short or noisy messages

4. **Dataset split**
   Split the dataset into:

   * `train.jsonl`
   * `val.jsonl`

5. **LoRA fine-tuning**
   Fine-tune a base language model (e.g. Phi-3, Mistral) to imitate the conversational style.

6. **Local inference**
   Run the fine-tuned model locally for interactive chat testing.

---

## 🚀 Features

* Convert Telegram JSON exports
* Convert WhatsApp chat exports
* Sanitize and clean dataset
* Train/validation dataset splitting
* LoRA fine-tuning for style imitation
* Local CLI chat testing
* Mixed RU + EN dataset support
* Apple GPU (MPS) acceleration

---

## 📦 Project Structure

```
project/
│
├── data/
│   ├── whatsapp/*.json
│   ├── telegram/*.json
│
├── dataset_build/
│   ├── convert_whatsapp_chat.py
│   ├── convert_telegram_chat.py
│   ├── sanitize_dataset.py
│   └── merge.py
│
├── split_train_val.py
├── train_qlora.py
├── chat_test.py
├── main.py
│
├── train.jsonl
├── val.jsonl
│
└── out_style_lora/
```

---

## ⚙ Installation

### 1. Create virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install torch transformers datasets peft trl accelerate
pip install certifi
```

---

## 📚 Dataset Preparation

### 1️⃣ Convert Telegram JSON

```bash
python convert_telegram_chat.py
```

### 2️⃣ Convert WhatsApp chat

```bash
python convert_whatsapp_chat.py
```

### 3️⃣ Merge datasets

```bash
python merge.py
```

### 4️⃣ Sanitize dataset

```bash
python sanitize_dataset.py
```

This step removes noise and improves training quality.

### 5️⃣ Split into train / validation

```bash
python split_train_val.py
```

Output:

```
train.jsonl
val.jsonl
```

---

## 🧠 LoRA Training

```bash
python train_qlora.py
```

Model output:

```
out_style_lora/final
```

---

## 💬 Local Chat Testing

```bash
python chat_test.py
```

Exit command:

```
/exit
```

---

## 🧩 Common Issues

### SSL certificate error

```bash
pip install -U certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

---

### High memory usage

Reduce generation length:

```python
max_new_tokens = 40
```

Or reduce training sequence length:

```
MAX_SEQ_LEN = 512
```

---

## 📈 Training Recommendations

* 2–4 epochs is usually sufficient
* train_loss ~ 0.6–0.9 is typical
* too many epochs may cause overfitting
* ensure dataset is clean before training

---

## 🔒 Privacy Notice

This model is trained on personal conversations.

* Do not publish raw datasets
* Remove sensitive information before training
* Use responsibly

---

## 🧪 Possible Improvements

* Larger context window
* Better dataset filtering
* Response length control
* Evaluation metrics
* Web UI
* Quantization for lower memory usage

---

## 🛠 Requirements

* Python 3.10+
* macOS Apple Silicon recommended
* 16GB RAM minimum recommended

---

This project provides a complete pipeline for building a personalized conversational style model from raw chat data to local inference.
