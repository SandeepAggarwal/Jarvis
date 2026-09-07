import argparse
from pathlib import Path
from typing import TypedDict

from rich.console import Console
from rich.panel import Panel
from local_agent import LocalAgent

console = Console()

WORKSPACE = Path("./workspace").resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = WORKSPACE / "local_agent_report.md"

class AgentState(TypedDict):
    task: str
    final_answer: str
    output_file: str

def call_local_model(user_prompt: str) -> str:
    agent = LocalAgent()
    return agent.run(user_prompt)

def synthesize_answer(state: AgentState) -> AgentState:
    console.rule("Step 4 - Local MLX Model Synthesis")

    user_prompt = """
User task:
{task}

After achieving the above task, please provide a final answer in the following format:
Create the final answer with these sections:
1. Quick Recommendation
2. Search-Based Context
3. Practical Next Steps
4. Tool Trace
""".format(
        task=state["task"]
    )

    try:
        answer = call_local_model(user_prompt)
    except Exception as error:
        answer = (
            "The external tools ran, but local model synthesis failed.\n\n"
            f"Error: {type(error).__name__}: {error}\n\n"
        )

    console.print(Panel(answer, title="Final Local Model Answer"))
    return {**state, "final_answer": answer}

def save_report(state: AgentState) -> AgentState:
    console.rule("Step 5 - Save Report")

    report_lines = [
        "# Local Agent Report",

        "",
        "## User Task",
        state.get("task", ""),
        "",

        "",
        "## Final Answer",
        state.get("final_answer", ""),
        "",
    ]

    report = "\n".join(report_lines)
    OUTPUT_FILE.write_text(report, encoding="utf-8")

    console.print(Panel(str(OUTPUT_FILE), title="Saved Report"))
    return {**state, "output_file": str(OUTPUT_FILE)}

def main():
    parser = argparse.ArgumentParser(description="Local MLX tool routing demo")
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    initial_state: AgentState = {
        "task": args.task,
        "final_answer": "",
        "output_file": "",
    }

    console.print(Panel(args.task, title="User Task"))

    final_state = synthesize_answer(initial_state)
    save_report(final_state)

    console.rule("Done")
    console.print(Panel(final_state["final_answer"], title="Final Answer"))
    console.print(f"Saved report: {final_state['output_file']}")

if __name__ == "__main__":
    main()
