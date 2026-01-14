#!/usr/bin/env python3
import argparse
from pathlib import Path

TEMPLATE_PATH = Path("src/prompt_templates/prompt_build_data.txt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True)
    args = parser.parse_args()

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    template = template.replace("{source_data_path}", "./source_data")
    template = template.replace("{output_path}", "./training_data")

    if args.agent in ("claude", "gemini"):
        template += (
            "\n\n"
            "You are running in a non-interactive mode. "
            "Make sure all processes finish and the final JSONL files are written before your last message.\n"
        )

    print(template)

if __name__ == '__main__':
    main()
