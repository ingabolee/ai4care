"""
cli.py -- Interactive terminal REPL for the AI4CARE interview engine.
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
    print("---")


def print_rag(rag: list):
    if rag:
        print("\n[RAG context]")
        for i, item in enumerate(rag, 1):
            print(f"  {i}. {item['text'][:120]} (score: {item['score']})")


def run(model_id: str):
    cfg = MODEL_CONFIGS.get(model_id, {})
    print(f"\nAI4CARE CLI | model: {cfg.get('local', model_id)}")
    print(f"             {cfg.get('description', '')}")
    print("Commands: /model <id>  /schema  /chroma  /exit\n")

    runner = get_runner(model_id, verbose=False)
    session = InterviewSession(episode_id="cli_0", model_id=model_id, runner=runner)
    print("Interviewer:", session.opening_question, "\n")

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

        else:
            try:
                state = session.respond(user_input)
            except RuntimeError as e:
                print(f"\n[Error] {e}")
                print("Exiting.")
                break

            for turn in reversed(state["turns"]):
                if turn["role"] == "interviewer" and turn.get("turn_id", 0) > 0:
                    print(f"\nInterviewer: {turn['text']}")
                    print_rag(turn.get("rag", []))
                    break
                elif turn["role"] == "system":
                    print(f"\n[System] {turn['text']}")
                    break

            completeness = state.get("completeness", 0)
            print(f"  [completeness: {completeness:.0%} | turn {state['turn_count']}/{15}]")

            if state["is_done"]:
                st = state
                if st["storage_success"]:
                    print(f"\nInterview complete. Artifact stored: {st['artifact_id']}")
                else:
                    print(f"\nInterview complete. Storage error: {st['storage_error']}")
                break

        print()


def main():
    parser = argparse.ArgumentParser(description="AI4CARE CLI Interview Tool")
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

    run(args.model)


if __name__ == "__main__":
    main()
