"""Realiza a validação dos dados convertidos do MedQuAD."""

import json
from pathlib import Path

def count_duplicates(dataset_path: Path) -> tuple[int, int]:
    """Conta o número de duplicatas no dataset."""
    seen = set()

    total_records = 0
    duplicate_recods = 0

    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:

            record = json.loads(line)

            question = record.get("instruction")
            answer = record.get("response")

            pair = (question, answer)

            total_records += 1

            if pair in seen:
                duplicate_recods += 1
            else:
                seen.add(pair)

    return total_records, duplicate_recods

def remove_duplicates(input_path: Path,output_path: Path) -> int:
    seen = set()
    unique_records = 0

    with input_path.open("r", encoding="utf-8") as source:
        with output_path.open("w", encoding="utf-8") as destination:
            for line in source:
                record = json.loads(line)

                pair = (
                    record["instruction"],
                    record["response"]
                )

                if pair in seen:
                    continue

                seen.add(pair)

                destination.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                unique_records += 1

    return unique_records

def main():
    """Realiza a validação do dataset MedQuAD convertido."""
    dataset_path = Path("data/processed/MedQuAD.jsonl")
    output_path = Path("data/processed/medquad_deduplicated.jsonl")

    total, duplicates = count_duplicates(dataset_path)
    unique_records = remove_duplicates(dataset_path, output_path)

    print(f"Total de registros: {total}")
    print(f"Total de duplicatas: {duplicates}")
    print(f"Total de registros únicos: {total - duplicates}")
    print(f"Registros exportados após limpeza: {unique_records}")

if __name__ == "__main__":
    main()