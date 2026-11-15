"""Lista todos os arquivos XML em um diretório."""

from pathlib import Path
from explore_medquad import read_qa_pairs

def count_xml_files(dataset_path: Path) -> int:
    total = 0

    """Conta o número de arquivos XML em um diretório e subdiretórios."""
    for xml_file in dataset_path.rglob("*.xml"):
        total += 1

    return total

def count_qa_pairs(dataset_path: Path) -> int:
    total = 0

    """Conta o número total de pares de perguntas e respostas em arquivos XML."""
    for xml_file in dataset_path.rglob("*.xml"):
        pairs = read_qa_pairs(xml_file)
        total += len(pairs)

    return total

def main():
    dataset_path = Path("data/raw/MedQuAD")

    total_arquivos = count_xml_files(dataset_path)
    total_pares = count_qa_pairs(dataset_path)
    
    print(f"\nTotal de arquivos XML encontrados: {total_arquivos}")
    print(f"Total de pares de perguntas e respostas encontrados: {total_pares}")

if __name__ == "__main__":
    main()