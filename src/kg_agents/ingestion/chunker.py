import os
import re
import json
from pathlib import Path
from typing import List, Dict


def extract_document_body(latex_text: str) -> str:
    match = re.search(r"\\begin{document}(.*)\\end{document}", latex_text, re.DOTALL)
    if match:
        return match.group(1)
    return latex_text


def split_by_section(text: str) -> List[Dict]:
    section_pattern = r"(\\section\*?{.*?})"
    parts = re.split(section_pattern, text)

    chunks = []
    current_heading = "Unknown"

    for part in parts:
        if part.startswith("\\section"):
            title_match = re.search(r"{(.*?)}", part)
            if title_match:
                current_heading = title_match.group(1).strip()
        else:
            stripped = part.strip()
            # Skip truly empty blocks — avoids IndexError downstream
            if stripped:
                chunks.append({"heading": current_heading, "content": stripped})

    return chunks


def paragraph_split(content: str, max_chars: int = 4000) -> List[str]:
    paragraphs = content.split("\n\n")

    final_chunks = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) < max_chars:
            buffer += para + "\n\n"
        else:
            if buffer.strip():
                final_chunks.append(buffer.strip())
            buffer = para + "\n\n"

    if buffer.strip():
        final_chunks.append(buffer.strip())

    return final_chunks


def chunk_latex_document(
    latex_path: str, output_dir: str, max_chars: int = 4000
) -> List[Dict]:

    if not os.path.exists(latex_path):
        raise FileNotFoundError(f"{latex_path} not found.")

    os.makedirs(output_dir, exist_ok=True)

    with open(latex_path, "r", encoding="utf-8") as f:
        latex_text = f.read()

    latex_text = extract_document_body(latex_text)

    section_blocks = split_by_section(latex_text)

    all_chunks = []
    doc_name = Path(latex_path).stem
    primary_counter = 0

    for block in section_blocks:

        sub_chunks = paragraph_split(block["content"], max_chars=max_chars)

        if not sub_chunks:
            primary_counter += 1
            continue

        if len(sub_chunks) == 1:
            all_chunks.append({
                "chunk_id": f"{doc_name}_chunk_{primary_counter}",
                "doc_id": doc_name,
                "heading": block["heading"],
                "content": sub_chunks[0].strip(),
            })
        else:
            for sub_index, sub_chunk in enumerate(sub_chunks, start=1):
                all_chunks.append({
                    "chunk_id": f"{doc_name}_chunk_{primary_counter}.{sub_index}",
                    "doc_id": doc_name,
                    "heading": block["heading"],
                    "content": sub_chunk.strip(),
                })

        primary_counter += 1

    output_path = os.path.join(output_dir, f"{doc_name}_chunks.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Chunks saved to {output_path} ({len(all_chunks)} chunks)")

    return all_chunks