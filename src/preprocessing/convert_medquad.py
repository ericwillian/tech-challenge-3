"""Converte pares de perguntas e respostas do MedQuAD em registros de dicionário."""

import json
from pathlib import Path
from explore_medquad import read_qa_pairs

def convert_pair_to_record(question: str, answer: str) -> dict[str, str]:
    """Converte um par de pergunta e resposta em um registro de dicionário."""
    record = {
        "instruction": question,
        "response": answer,
    }

    return record

def main():
    """Realiza a conversão dos pares de perguntas e respostas em um arquivo JSONL."""
    dataset_path = Path("data/raw/MedQuAD")
    output_path = Path("data/processed/MedQuAD.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for xml_file in dataset_path.rglob("*.xml"):
            pairs = read_qa_pairs(xml_file)

            for question, answer in pairs:
                record = convert_pair_to_record(question, answer)

                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                total += 1

    print(f"Conversão concluída. Total de pares convertidos: {total}")

if __name__ == "__main__":
    main()