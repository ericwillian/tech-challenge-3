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
            record["instruction"].casefold().split()
        )
        groups[question].append(record)

    return list(groups.values())

def split_groups(groups: list[list[dict]]) -> tuple[list, list, list]:
    """Divide os grupos em conjuntos de treino, validação e teste."""
    random.seed(42) # Para reprodutibilidade
    random.shuffle(groups)

    total = len(groups)

    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train_groups = groups[:train_end]
    val_groups = groups[train_end:val_end]
    test_groups = groups[val_end:]

    return train_groups, val_groups, test_groups

def save_dataset(groups: list[list[dict]], output_path: Path) -> int:
    """Salva os grupos em um arquivo JSONL e retorna o número de registros salvos."""
    total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for group in groups:
            for record in group:
                file.write(
                     json.dumps(record, ensure_ascii=False) + "\n"
                )
                total += 1

    return total

def main():
    """Função principal para carregar registros e agrupá-los por pergunta."""
    dataset_path = Path("data/processed/medquad_deduplicated.jsonl")
    output_dir = Path("data/processed")

    records = load_records(dataset_path)
    groups = group_by_question(records)

    train, val, test = split_groups(groups)

    train_count = save_dataset(train, output_dir / "train.jsonl")
    val_count = save_dataset(val, output_dir / "val.jsonl")
    test_count = save_dataset(test, output_dir / "test.jsonl")

    print(f"Treinamento: {train_count}")
    print(f"Validação: {val_count}")
    print(f"Teste: {test_count}")
    print(f"Total: {len(records)}")

if __name__ == "__main__":
    main()