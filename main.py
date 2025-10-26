#!/usr/bin/env python3

import argparse
from rich import print
from rag_pipeline.pipeline1 import RAGPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild FAISS+SQLite index from PDFs in data/",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="If set, summarize answer with Gemini 2.5 Flash in <=50 words.",
    )
    args = parser.parse_args()

    rag = RAGPipeline()
    rag.build_index(rebuild=args.rebuild)

    print("[bold green]RAG index ready.[/bold green]")
    print("Ask me things about your PDFs. Type 'exit' to quit.\n")

    while True:
        q = input("❓ Question > ").strip()
        if q.lower() in ("exit", "quit"):
            break

        result = rag.ask(q, use_llm=args.llm)

        print("\n[bold cyan]--- ANSWER PACKAGE ---[/bold cyan]")
        print(f"[bold]Question:[/bold] {result.question}\n")
        print(f"[bold]Answer:[/bold]\n{result.answer}\n")

        print("[bold]Context chunks:[/bold]")
        for hit in result.context_chunks:
            print(
                f"- score={hit.score:.3f} | {hit.chunk.doc_id} "
                f"chars {hit.chunk.start_char}-{hit.chunk.end_char}"
            )

            preview_text = hit.chunk.text[:300].replace("\n", " ")
            print(f"  {preview_text}...\n")

        print("[bold cyan]-----------------------[/bold cyan]\n")

    rag.close()

if __name__ == "__main__":
    main()
