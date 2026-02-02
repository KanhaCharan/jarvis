"""Simple Jar CLI entry point."""

from core import JarCore


def main() -> None:
    core = JarCore()
    print("Jarvis: Hi - I'm Jarvis, your personal AI assistant. Type 'exit' to quit.")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nJarvis: Goodbye.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Jarvis: Goodbye.")
            break

        result = core.handle(user_text)
        if result is not None:
            print(f"Jarvis: {result}")


if __name__ == "__main__":
    main()
