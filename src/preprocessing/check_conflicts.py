"""Verifica se há conflitos entre os dados de entrada, garantindo que não haja duplicatas ou inconsistências."""

import json
from pathlib import Path

def find_repeated_questions(dataset_path: Path) -> dict[str, set[str]]:
    """Encontra perguntas repetidas no conjunto de dados de entrada."""
    questions = {}

    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            question = record["instruction"]
            answer = record["response"]

            if question not in questions:
                questions[question] = set()

            questions[question].add(answer)

    return {
        question: answers
        for question, answers in questions.items()
        if len(answers) > 1
    }

def main():
    """Função principal para verificar conflitos no conjunto de dados."""
    dataset_path = Path("data/processed/medquad_deduplicated.jsonl")

    repeated = find_repeated_questions(dataset_path)

    print(f"Perguntas com múltiplas respostas: {len(repeated)}")

    for question, answers in list(repeated.items())[:3]:  # Exibe apenas as primeiras 3 perguntas com múltiplas respostas
        print(f"\nPergunta: {question}")
        print(f"Quantidade de respostas: {len(answers)}")

        for index, answer in enumerate(sorted(answers), start=1):
            print(f"\nResposta {index}: {answer[:500]}")

if __name__ == "__main__":
    main()
