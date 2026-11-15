"""Realiza a checagem de qualidade dos dados de entrada, garantindo que estejam no formato esperado e livres de inconsistências."""
import json
from pathlib import Path

def check_quality(input_path: Path) -> tuple[int, int]:
    """Verifica a qualidade dos dados de entrada, contando o número de registros válidos e inválidos."""
    valid_records = 0
    invalid_records = 0

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            question = record.get("instruction", "")
            answer = record.get("response", "")

            if (
                isinstance(question, str)
                and isinstance(answer, str)
                and question.strip()
                and answer.strip()
            ):
                valid_records += 1
            else:
                invalid_records += 1

    return valid_records, invalid_records

def main():
    dataset_path = Path("data/processed/medquad_deduplicated.jsonl")

    valid, invalid = check_quality(dataset_path)

    print(f"Registros válidos: {valid}")
    print(f"Registros inválidos: {invalid}")

if __name__ == "__main__":
    main()