import argparse
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from rag_pipeline.pipeline1 import RAGPipeline
from rag_pipeline.logger import get_logger

console = Console()
log = get_logger()


def print_answer(answer_text: str):
    console.print(Panel.fit(answer_text, title="Answer", border_style="green"))


def print_context_chunks(hits):
    if not hits:
        console.print("[bold red]No context chunks found.[/bold red]")
        return

    console.print(Panel.fit("Context chunks", border_style="blue"))

    for idx, hit in enumerate(hits[:12], start=1):  # limit to top 12 for readability
        score_str = f"{hit.score:.3f}" if hit.score is not None else "-"
        source_str = hit.chunk.doc_id
        span_str = f"{hit.chunk.start_char}-{hit.chunk.end_char}"

        preview_text = hit.chunk.text[:300].replace("\n", " ")
        preview_str = preview_text + "..."

        console.print(f"[bold cyan]{idx}.[/bold cyan]")
        console.print(f"[bold]score:[/bold] {score_str}")
        console.print(f"[bold]source:[/bold] {source_str}")
        console.print(f"[bold]span:[/bold] {span_str}")
        console.print(f"[bold]preview:[/bold] {preview_str}\n")


def interactive_loop(rag: RAGPipeline, use_llm: bool):
    console.print("[bold]Interactive mode. Type 'exit' or 'quit' to stop.[/bold]")

    while True:
        question = Prompt.ask("\n[bold cyan]❓ Question[/bold cyan]").strip()
        if question.lower() in ("exit", "quit"):
            console.print("[bold red]Bye.[/bold red]")
            log.info("Session ended by user.")
            break

        log.info("User question: %r (use_llm=%s)", question, use_llm)

        pkg = rag.ask(question, use_llm=use_llm)

        print_answer(pkg.answer)
        print_context_chunks(pkg.context_chunks)

        log.info("Answer preview: %r", pkg.answer[:200])
        log.debug("Returned %d context chunks", len(pkg.context_chunks))


def main():
    parser = argparse.ArgumentParser(
        description="RAG pipeline CLI (PDF / DOCX / TXT + FAISS + Gemini summarizer)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector store and SQLite index from data/ (required after adding/changing files)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use Gemini 2.5 Flash to generate a concise ≤50-word bullet answer",
    )

    args = parser.parse_args()

    log.info("Starting RAG app. rebuild=%s llm=%s", args.rebuild, args.llm)

    rag = RAGPipeline()

    try:
        rag.build_index(rebuild=args.rebuild)
        interactive_loop(rag, use_llm=args.llm)
    except Exception as e:
        log.exception("Fatal error in main loop: %s", e)
        raise
    finally:
        rag.close()
        log.info("RAG app shutdown complete.")


if __name__ == "__main__":
    main()
