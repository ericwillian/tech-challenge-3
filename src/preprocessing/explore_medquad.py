"""Inspeciona pares de perguntas e respostas em um XML local do MedQuAD."""

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def extract_text(element: ET.Element | None) -> str:
    """Extrai texto, incluindo elementos internos, e normaliza espacos."""
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def read_qa_pairs(xml_path: Path) -> list[tuple[str, str]]:
    """Le o XML e retorna pares QAPair com pergunta e resposta nao vazias."""
    root = ET.parse(xml_path).getroot()
    pairs = []
    for pair in root.iter("QAPair"):
        question = extract_text(pair.find("Question"))
        answer = extract_text(pair.find("Answer"))
        if question and answer:
            pairs.append((question, answer))
    return pairs


def main() -> int:
    """Processa o argumento, apresenta os pares e trata erros de leitura."""
    parser = argparse.ArgumentParser(
        description="Explora perguntas e respostas de um arquivo XML do MedQuAD."
    )
    parser.add_argument("xml_path", type=Path, help="Caminho do arquivo XML local")
    args = parser.parse_args()

    try:
        pairs = read_qa_pairs(args.xml_path)
    except FileNotFoundError:
        print(f"Erro: arquivo nao encontrado: {args.xml_path}", file=sys.stderr)
        return 1
    except ET.ParseError as error:
        print(f"Erro: XML invalido: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Erro ao ler o arquivo: {error}", file=sys.stderr)
        return 1

    print(f"Quantidade de pares encontrados: {len(pairs)}")
    for index, (question, answer) in enumerate(pairs[:3], start=1):
        print(f"\nExemplo {index}")
        print(f"Pergunta: {question}")
        print(f"Resposta: {answer[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
