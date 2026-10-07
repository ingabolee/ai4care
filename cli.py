"""
cli.py -- Interactive terminal REPL for the AI4CARE interview engine.
<<<<<<< HEAD
          Updated to use the single Chroma collection and model registry.

Run:
    python cli.py [--model MODEL_ID]
    python cli.py --list-models
"""

import argparse
import sys

from model_registry import MODEL_CONFIGS, get_enabled_models, get_runner
from engine import InterviewSession


def print_schema(knowledge_state: dict, completeness: float):
    print(f"\n--- Knowledge State ({completeness:.0%} complete) ---")
    for fname, fstate in knowledge_state.items():
        val = fstate["value"] or "pending"
        status = fstate["status"]
        print(f"  {fname}: [{status}] {str(val)[:80]}")
=======
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
>>>>>>> 663c32fa4d020ebf24239f0135157ae04f1562a9
    print("---")


def print_rag(rag: list):
    if rag:
        print("\n[RAG context]")
        for i, item in enumerate(rag, 1):
            print(f"  {i}. {item['text'][:120]} (score: {item['score']})")


def run(model_id: str):
<<<<<<< HEAD
    cfg = MODEL_CONFIGS.get(model_id, {})
    print(f"\nAI4CARE CLI | model: {cfg.get('local', model_id)}")
    print(f"             {cfg.get('description', '')}")
    print("Commands: /model <id>  /schema  /chroma  /exit\n")

    runner = get_runner(model_id, verbose=False)
    session = InterviewSession(episode_id="cli_0", model_id=model_id, runner=runner)
    print("Interviewer:", session.opening_question, "\n")
=======
    print(f"\nAI4CARE CLI | model: {MODELS[model_id]['local']}")
    print("Commands: /model <id>  /schema  /chroma  /exit\n")

    session = InterviewSession(model_id=model_id)
    print("Interviewer:", session.turns[0]["text"], "\n")
>>>>>>> 663c32fa4d020ebf24239f0135157ae04f1562a9

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
<<<<<<< HEAD
            if len(parts) == 2 and parts[1] in MODEL_CONFIGS:
                runner = get_runner(parts[1], verbose=False)
                session = InterviewSession(episode_id="cli_1", model_id=parts[1], runner=runner)
                print(f"Switched to model: {parts[1]}")
            else:
                print("Available models:", ", ".join(MODEL_CONFIGS.keys()))

        elif user_input == "/schema":
            state = session._state()
            print_schema(state["knowledge_state"], state["completeness"])

        elif user_input == "/chroma":
            from knowledge import inspect
            info = inspect()
            print(f"  Collection: {info['collection']}  |  Documents: {info['total_documents']}")
=======
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
>>>>>>> 663c32fa4d020ebf24239f0135157ae04f1562a9

        else:
            try:
                state = session.respond(user_input)
            except RuntimeError as e:
                print(f"\n[Error] {e}")
                print("Exiting.")
                break

            for turn in reversed(state["turns"]):
<<<<<<< HEAD
                if turn["role"] == "interviewer" and turn.get("turn_id", 0) > 0:
=======
                if turn["role"] == "interviewer":
>>>>>>> 663c32fa4d020ebf24239f0135157ae04f1562a9
                    print(f"\nInterviewer: {turn['text']}")
                    print_rag(turn.get("rag", []))
                    break
                elif turn["role"] == "system":
                    print(f"\n[System] {turn['text']}")
                    break

<<<<<<< HEAD
            completeness = state.get("completeness", 0)
            print(f"  [completeness: {completeness:.0%} | turn {state['turn_count']}/{15}]")

            if state["is_done"]:
                st = state
                if st["storage_success"]:
                    print(f"\nInterview complete. Artifact stored: {st['artifact_id']}")
                else:
                    print(f"\nInterview complete. Storage error: {st['storage_error']}")
=======
            print(f"  [schema: {state['filled']}/9 fields filled | turn {state['turn_count']}/{15}]")

            if state["is_done"]:
                print("\nInterview complete. Knowledge artifact stored in ChromaDB.")
>>>>>>> 663c32fa4d020ebf24239f0135157ae04f1562a9
                break

        print()


def main():
    parser = argparse.ArgumentParser(description="AI4CARE CLI Interview Tool")
<<<<<<< HEAD
    parser.add_argument(
        "--model", default="general_small",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use for interview generation",
    )
    parser.add_argument("--list-models", action="store_true", help="List all registered models and exit")
    args = parser.parse_args()

    if args.list_models:
        print("\nRegistered Models:")
        for mid, cfg in MODEL_CONFIGS.items():
            status = "[enabled]" if cfg.get("enabled") else "[disabled]"
            print(f"  {status} {mid:<20} {cfg.get('description', '')}")
        return

=======
    parser.add_argument("--model", default="general_small", choices=list(MODELS.keys()),
                        help="Model to use for interview generation")
    args = parser.parse_args()
>>>>>>> 663c32fa4d020ebf24239f0135157ae04f1562a9
    run(args.model)


if __name__ == "__main__":
    main()
