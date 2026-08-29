"""
cli.py -- Interactive terminal REPL for the AI4CARE interview engine.
Run: python cli.py [--model general_small|general_main|coder_small|coder_main]
"""

import sys
import argparse
from setup import MODELS
from engine import InterviewSession


def print_schema(schema: dict, filled: int):
    print(f"\n--- Mind Map ({filled}/9 fields) ---")
    for key, val in schema.items():
        display = ", ".join(val) if isinstance(val, list) else val
        print(f"  {key}: {display or 'pending'}")
    print("---")


def print_rag(rag: list):
    if rag:
        print("\n[RAG context]")
        for i, item in enumerate(rag, 1):
            print(f"  {i}. {item['text'][:120]} (score: {item['score']})")


def run(model_id: str):
    print(f"\nAI4CARE CLI | model: {MODELS[model_id]['local']}")
    print("Commands: /model <id>  /schema  /chroma  /exit\n")

    session = InterviewSession(model_id=model_id)
    print("Interviewer:", session.turns[0]["text"], "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("Goodbye.")
            break

        elif user_input.startswith("/model"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1] in MODELS:
                session.model_id = parts[1]
                session._llm = None  # unload cached model
                print(f"Switched to model: {parts[1]}")
            else:
                print("Available models:", ", ".join(MODELS.keys()))

        elif user_input == "/schema":
            state = session._state()
            print_schema(state["schema"], state["filled"])

        elif user_input == "/chroma":
            from knowledge import inspect
            for model_id_k, info in inspect().items():
                print(f"  {info['collection']}: {info['count']} docs")

        else:
            try:
                state = session.respond(user_input)
            except RuntimeError as e:
                print(f"\n[Error] {e}")
                print("Exiting.")
                break

            for turn in reversed(state["turns"]):
                if turn["role"] == "interviewer":
                    print(f"\nInterviewer: {turn['text']}")
                    print_rag(turn.get("rag", []))
                    break
                elif turn["role"] == "system":
                    print(f"\n[System] {turn['text']}")
                    break

            print(f"  [schema: {state['filled']}/9 fields filled | turn {state['turn_count']}/{15}]")

            if state["is_done"]:
                print("\nInterview complete. Knowledge artifact stored in ChromaDB.")
                break

        print()


def main():
    parser = argparse.ArgumentParser(description="AI4CARE CLI Interview Tool")
    parser.add_argument("--model", default="general_small", choices=list(MODELS.keys()),
                        help="Model to use for interview generation")
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
