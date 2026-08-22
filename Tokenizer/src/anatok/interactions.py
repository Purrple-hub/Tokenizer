"""Interactive mode module for the Tokenizer project.

Handles interactive tokenization via command line input.
"""

import sys
from .utils import count_tokens, truncate_text, get_file_info

def interactive_mode():
    """Run the tokenizer in interactive mode."""
    print("Tokenizer Interactive Mode")
    print("=" * 40)
    print("Enter file paths or text to tokenize.")
    print("Type 'exit' or 'quit' to exit.")
    print()

    while True:
        try:
            user_input = input("Tokenizer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        import os
        if os.path.exists(user_input):
            info = get_file_info(user_input)
            print(f"\nFile: {user_input}")
            print(f"Size: {info.get('size', 'unknown')} bytes")

            from .tokenizer import tokenize_file
            result = tokenize_file(user_input)
            print(f"Tokens: {result['token_count']}")
            print(f"Formatted size: {result['formatted_size']}")

        else:
            preview = truncate_text(user_input, 200)
            token_count = count_tokens(user_input)
            print(f"\nText: {preview}")
            print(f"Token count: {token_count}")
