"""Realiza a verificação de vazamento de dados no conjunto de dados fornecido."""

import json
from pathlib import Path

def load_questions(dataset_path: Path) -> set[str]:
    """Carrega as perguntas do conjunto de dados a partir de um arquivo JSONL."""
    questions = set()

    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            
            question = " ".join(
                record["instruction"].casefold().split()
            )

            questions.add(question)

    return questions

def main():
    """Função principal para carregar perguntas e verificar vazamento de dados."""
    data_dir = Path("data/processed")

    train = load_questions(data_dir / "train.jsonl")
    val = load_questions(data_dir / "val.jsonl")
    test = load_questions(data_dir / "test.jsonl")

    train_val = train.intersection(val)
    train_test = train.intersection(test)
    val_test = val.intersection(test)

    print(f"Treinamento x Validação: {len(train_val)}")
    print(f"Treinamento x Teste: {len(train_test)}")
    print(f"Validação x Teste: {len(val_test)}")

    if not train_val and not train_test and not val_test:
        print("Nenhuma pergunta compartilhada entre os conjuntos de dados!")
    else:
        print("Atenção: Perguntas compartilhadas encontradas entre os conjuntos de dados!")

if __name__ == "__main__":
    main()