import os
import re
import json
from pathlib import Path
from typing import List, Dict


def extract_document_body(latex_text: str) -> str:
    """
    Extract content inside \\begin{document} ... \\end{document}
    """
    match = re.search(r"\\begin{document}(.*)\\end{document}", latex_text, re.DOTALL)
    if match:
        return match.group(1)
    return latex_text


def split_by_section(text: str) -> List[Dict]:
    """
    Sequentially split LaTeX document by \\section.
    Each block keeps nearest section heading.
    """

    section_pattern = r"(\\section\*?{.*?})"

    parts = re.split(section_pattern, text)

    chunks = []
    current_heading = "Unknown"

    for part in parts:
        # if not part.strip():
        #     continue

        if part.startswith("\\section"):
            # Extract heading title
            title_match = re.search(r"{(.*?)}", part)
            if title_match:
                current_heading = title_match.group(1).strip()
        else:
            chunks.append({"heading": current_heading, "content": part.strip()})

    return chunks


def paragraph_split(content: str, max_chars: int = 4000) -> List[str]:
    """
    Split content into size-controlled chunks
    without breaking equations (paragraph-based splitting).
    """

    paragraphs = content.split("\n\n")

    final_chunks = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) < max_chars:
            buffer += para + "\n\n"
        else:
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

    # Step 1: Split by section
    section_blocks = split_by_section(latex_text)

    all_chunks = []
    doc_name = Path(latex_path).parent.parent.name

    # Step 2: Split large blocks paragraph-wise
    primary_counter = 0

    for block in section_blocks:
        sub_chunks = paragraph_split(block["content"], max_chars=max_chars)

        # If only one chunk, keep clean numbering
        if len(sub_chunks) == 0:
            all_chunks.append(
                {
                    "chunk_id": f"{doc_name}_chunk_{primary_counter}",
                    "doc_id": doc_name,
                    "heading": block["heading"],
                    "content": sub_chunks[0].strip() if sub_chunks else "",
                }
            )
        elif len(sub_chunks) == 1:
            all_chunks.append(
                {
                    "chunk_id": f"{doc_name}_chunk_{primary_counter}",
                    "doc_id": doc_name,
                    "heading": block["heading"],
                    "content": sub_chunks[0].strip(),
                }
            )
        else:
            # Multiple splits → use 3.1, 3.2 style
            for sub_index, sub_chunk in enumerate(sub_chunks, start=1):
                all_chunks.append(
                    {
                        "chunk_id": f"{doc_name}_chunk_{primary_counter}.{sub_index}",
                        "doc_id": doc_name,
                        "heading": block["heading"],
                        "content": sub_chunk.strip(),
                    }
                )

        primary_counter += 1

    # Step 3: Save JSON
    output_path = os.path.join(output_dir, f"{doc_name}_chunks.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Chunks saved to {output_path}")

    return all_chunks
