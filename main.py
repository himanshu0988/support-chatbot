"""Command-line interface for the support chatbot."""
import argparse

from chatbot import DialogueManager

EXIT_WORDS = {"quit", "exit", "q"}


def main() -> None:
    parser = argparse.ArgumentParser(description="ShopEasy customer support chatbot")
    parser.add_argument("--debug", action="store_true", help="show intent, confidence and source")
    parser.add_argument("--no-llm", action="store_true", help="disable the LLM fallback")
    args = parser.parse_args()

    bot = DialogueManager(use_llm=not args.no_llm)
    print("ShopEasy Support Bot (type 'quit' to exit)")
    if bot.llm and bot.llm.available:
        print("[hybrid mode: LLM fallback enabled]")
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break
        if user.lower() in EXIT_WORDS:
            print("Bot: Goodbye! Thanks for contacting ShopEasy.")
            break
        r = bot.reply(user)
        print(f"Bot: {r.text}")
        if args.debug:
            print(f"     [intent={r.intent} conf={r.confidence:.2f} source={r.source}]")


if __name__ == "__main__":
    main()
