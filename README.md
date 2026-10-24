# ShopEasy Support Bot: Rule-Based NLP Customer Support Chatbot

A hybrid customer-support chatbot that understands natural-language queries, recognises intent and replies from a JSON knowledge base. It combines **rule-based logic**, **TF-IDF + cosine similarity**, and a **scikit-learn intent classifier**, with an optional **LLM fallback** for low-confidence queries.

**Stack:** Python · NLTK · scikit-learn · TF-IDF · JSON

## How it works

```
User message
   │
   ▼
Rules ──► order-ID slot filling, "talk to a human" detection, follow-up handling
   │
   ▼
Preprocessing: clean → tokenize → stop-word removal (keeps negations) → lemmatize
   │
   ├──► Intent classifier (word+char TF-IDF → Logistic Regression) → intent + probability
   └──► Cosine-similarity matcher (TF-IDF vs. every known pattern) → nearest pattern + score
   │
   ▼
Both confident / agree? ──► yes: reply from intents.json
   │ no
   ▼
LLM fallback (if ANTHROPIC_API_KEY set) ──► otherwise polite canned fallback
```

## Project structure

```
chatbot/
  preprocessing.py       # tokenization, stop words, lemmatization
  knowledge_base.py      # loads data/intents.json
  matcher.py             # TF-IDF + cosine similarity
  intent_classifier.py   # scikit-learn pipeline
  dialogue_manager.py    # rules + thresholds + fallback routing + slot memory
  llm_fallback.py        # optional hybrid LLM integration
data/intents.json        # 12 intents, ~25 phrases each, responses
tests/test_chatbot.py    # pytest suite
train_eval.py            # train + evaluate + save model and metrics
main.py                  # CLI chatbot
```

## Setup

```bash
git clone https://github.com/<himanshu0988>/support-chatbot.git
cd support-chatbot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_eval.py            # trains the model, prints metrics (NLTK data downloads automatically)
python main.py --debug          # chat; --debug shows intent/confidence/source
python -m pytest                # run tests
```

### Optional: hybrid LLM mode

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key"     # Windows PowerShell: $env:ANTHROPIC_API_KEY="your-key"
python main.py
```

When the rule-based layers are not confident, the query goes to an LLM with a customer-support system prompt and the last few turns of history. Without a key, the bot uses a canned fallback instead.

## Results

Measured on the 295-phrase intent set in `data/intents.json` (`python train_eval.py`):

| Metric | Value |
|---|---|
| Hold-out accuracy (75/25 stratified split) | ~86% |
| 5-fold cross-validated accuracy | ~84% |

These numbers come from a small, hand-written dataset, so they show the pipeline works rather than real-world performance. The main way to improve it is more varied training phrases (the first version with ~12 phrases per intent scored ~68%). Results will change slightly if you edit the data.

## Extending

Add a new entry to `data/intents.json` (`tag`, `patterns`, `responses`) and run `python train_eval.py`. No code changes needed.

## Future work

Context-aware multi-turn dialogue, real order-lookup API integration, web UI (Flask/Streamlit), and sentence embeddings in place of TF-IDF.

## License

MIT
