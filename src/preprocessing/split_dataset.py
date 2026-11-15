"""Realiza a separação do dataset em treino, validação e teste."""

import json
import random
from collections import defaultdict
from pathlib import Path

def load_records(dataset_path: Path) -> list[dict]:
    """Carrega os registros do conjunto de dados a partir de um arquivo JSONL."""
    records = []

    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records

def group_by_question(records: list[dict]) -> list[list[dict]]:
    """Agrupa os registros por pergunta, garantindo que cada grupo contenha todas as respostas para uma pergunta específica."""
    groups = defaultdict(list)

    for record in records:
        question = " ".join(
            record["instructions"].casefold().split()
        )
        groups[question].append(record)

    return list(groups.values())

def main() -> None:
    """Função principal para carregar registros e agrupá-los por pergunta."""
    dataset_path = Path("data/processed/medquad_deduplicated.jsonl")

    records = load_records(dataset_path)
    groups = group_by_question(records)

    print(f"Total de registros: {len(records)}")
    print(f"Total de grupos: {len(groups)}")

    grouped_records = sum(len(group) for group in groups)

    print(f"Registros após agrupamento: {grouped_records}")


if __name__ == "__main__":
    main()