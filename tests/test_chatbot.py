import pytest

from chatbot import DialogueManager
from chatbot.preprocessing import clean_text, preprocess, preprocess_tokens


@pytest.fixture(scope="module")
def bot():
    return DialogueManager(use_llm=False)


def test_clean_text():
    assert clean_text("  Hello,   World!! ") == "hello world"
    assert "not" in clean_text("I can't login")


def test_preprocess_removes_stopwords_and_lemmatises():
    tokens = preprocess_tokens("The orders are arriving")
    assert "the" not in tokens and "order" in tokens


def test_preprocess_keeps_negation():
    assert "not" in preprocess("I do not want it")


@pytest.mark.parametrize("msg,intent", [
    ("hello", "greeting"),
    ("where is my package", "order_status"),
    ("I want to cancel my purchase", "cancel_order"),
    ("what is your refund policy", "return_refund"),
    ("how many days will delivery take", "shipping_info"),
    ("do you accept paypal", "payment_methods"),
    ("I forgot my password", "account_help"),
    ("my coupon code is not working", "discount_coupon"),
    ("bye", "goodbye"),
])
def test_intents(msg, intent):
    assert DialogueManager(use_llm=False).reply(msg).intent == intent


def test_human_rule(bot):
    r = bot.reply("let me speak to a real person")
    assert r.intent == "human_agent" and r.source == "rule"


def test_order_id_slot_filling():
    b = DialogueManager(use_llm=False)
    b.reply("where is my order")
    r = b.reply("#123456")
    assert r.intent == "order_status"
    assert b.order_id == "123456"


def test_gibberish_falls_back(bot):
    r = bot.reply("zxqv blorp wibble")
    assert r.source == "fallback" and r.intent is None


def test_empty_message(bot):
    assert bot.reply("   ").source == "rule"
