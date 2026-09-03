"""
ingest_mac.py

Loads downloaded macOS technical documentation from data/mac_raw/
and converts it into structured document records.

Output:
    data/mac_processed/documents.json
"""

import json
from pathlib import Path


RAW_DIR = Path("data/mac_raw")

OUTPUT_FILE = Path(
    "data/mac_processed/documents.json"
)


def parse_mac_document(
    file_path: Path,
) -> dict | None:

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if not text.strip():
        return None

    metadata = {}

    body_lines = []
    parsing_header = True

    for line in text.splitlines():

        if parsing_header and not line.strip():
            parsing_header = False
            continue

        if parsing_header and ":" in line:
            key, value = line.split(":", 1)

            metadata[
                key.strip().lower()
            ] = value.strip()

        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()

    if not body:
        return None

    try:
        authority_score = float(
            metadata.get(
                "authority-score",
                0.5,
            )
        )

    except ValueError:
        authority_score = 0.5

    return {
        "source": file_path.name,
        "page": 1,

        "source_url":
            metadata.get("source-url"),

        "publisher":
            metadata.get(
                "publisher",
                "Unknown",
            ),

        "topic":
            metadata.get(
                "topic",
                "macos",
            ),

        "authority":
            metadata.get(
                "authority",
                "unknown",
            ),

        "authority_score":
            authority_score,

        "knowledge_type":
            "mac_knowledge",

        "text": body,
    }


def load_all_documents() -> list[dict]:

    documents = []

    files = sorted(
        RAW_DIR.glob("*.txt")
    )

    if not files:
        print(
            f"No files found in "
            f"{RAW_DIR}/"
        )

        return documents

    for file_path in files:

        try:
            doc = parse_mac_document(
                file_path
            )

            if doc:
                documents.append(doc)

                print(
                    f"Loaded "
                    f"{file_path.name} "
                    f"[{doc['topic']}] "
                    f"[{doc['authority']}]"
                )

        except Exception as exc:
            print(
                f"Failed loading "
                f"{file_path.name}: "
                f"{exc}"
            )

    return documents


if __name__ == "__main__":

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    documents = load_all_documents()

    OUTPUT_FILE.write_text(
        json.dumps(
            documents,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved {len(documents)} "
        f"Mac documents to "
        f"{OUTPUT_FILE}"
    )