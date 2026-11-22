
# Relatório Técnico — Tech Challenge Fase 3

## 1. Introdução

Este projeto apresenta o desenvolvimento de um protótipo acadêmico de assistente virtual médico, utilizando técnicas de Fine-Tuning de modelos de linguagem, Retrieval-Augmented Generation (RAG) e orquestração de agentes com LangGraph.

O objetivo é demonstrar como um modelo de linguagem personalizado pode ser integrado a informações de prontuários hospitalares, permitindo responder perguntas médicas gerais e consultar registros clínicos contextualizados.

O sistema utiliza dados públicos de perguntas e respostas médicas e prontuários inteiramente sintéticos. Sua finalidade é exclusivamente educacional, não sendo adequado para decisões clínicas reais.

## 2. Dataset e preparação dos dados

### 2.1. Dataset MedQuAD

Para o Fine-Tuning, foi utilizado o **MedQuAD**, um conjunto público de perguntas e respostas relacionadas à saúde.

Os dados passaram por etapas de extração, normalização, remoção de duplicatas, formatação e separação entre treinamento, validação e teste.

O conjunto de teste foi reservado para avaliação independente, sem participação no treinamento.

Para o treinamento ampliado, foram selecionados exemplos que respeitavam o limite máximo de 512 tokens, evitando truncamentos que poderiam comprometer as respostas.

Os conjuntos elegíveis resultaram em:

| Conjunto    | Exemplos |
| ----------- | -------: |
| Treinamento |   11.391 |
| Validação |    1.439 |

### 2.2. Dados sintéticos hospitalares

Para demonstrar o agente de consulta aos prontuários, foram gerados **1.000 pacientes sintéticos**, contendo dados cadastrais, observações clínicas e exames fictícios.

Cada paciente possui um identificador interno, como `PAC-DEMO-001`, utilizado para recuperar os documentos correspondentes.

Os dados foram organizados em **3.000 documentos** no ChromaDB, distribuídos entre cadastros, anotações clínicas e exames.

Os registros são fictícios e não representam pacientes reais.

## 3. Processo de Fine-Tuning

### 3.1. Modelo utilizado

O modelo selecionado foi o **Qwen2.5-0.5B-Instruct**, disponibilizado pela Hugging Face.

Optou-se pelo **Full Fine-Tuning**, no qual os pesos treináveis do modelo são atualizados. Diferentemente do LoRA, não foram utilizados apenas adaptadores de baixa dimensão.

### 3.2. Preparação para treinamento

Os exemplos foram convertidos para o formato de conversação esperado pelo modelo. A tokenização incluiu máscaras para restringir o cálculo da perda aos tokens apropriados da resposta.

Foi realizado inicialmente um experimento reduzido, utilizado para verificar o funcionamento do pipeline. Posteriormente, iniciou-se o treinamento ampliado a partir dos pesos originais do Qwen.

### 3.3. Configuração do experimento principal

| Parâmetro                          | Configuração                |
| ----------------------------------- | ----------------------------- |
| Modelo                              | Qwen2.5-0.5B-Instruct         |
| Técnica                            | Full Fine-Tuning              |
| Épocas                             | 1                             |
| Taxa de aprendizado                 | 2e-5                          |
| Batch size por dispositivo          | 1                             |
| Acumulação de gradientes          | 8                             |
| Otimizador                          | AdamW                         |
| Precisão dos pesos carregados      | FP32                          |
| Precisão mista durante treinamento | FP16                          |
| Gradient checkpointing              | Habilitado                    |
| Ambiente de treinamento             | Google Colab com GPU Tesla T4 |

Os pesos completos e o tokenizer foram salvos ao término do treinamento e exportados para posterior utilização em inferência.

## 4. Resultados do treinamento

O treinamento ampliado foi concluído após aproximadamente **1 hora e 2 minutos**, totalizando 1.424 etapas de otimização.

| Métrica               |         Resultado |
| ---------------------- | ----------------: |
| Tempo de treinamento   | 3.735,32 segundos |
| Train loss final       |           1,28149 |
| Eval loss final        |           1,21580 |
| Épocas concluídas    |                 1 |
| Etapas de otimização |             1.424 |

A execução demonstrou que o pipeline de treinamento completo conseguiu processar os exemplos elegíveis e salvar os pesos ajustados.

Entretanto, valores de loss não representam diretamente acurácia médica ou taxa de respostas corretas. Uma avaliação clínica sistemática exigiria métricas adicionais e revisão especializada.

## 5. Arquitetura do assistente médico

O assistente foi desenvolvido com duas rotas principais de processamento, coordenadas pelo **LangGraph**.

### 5.1. Agente de Fine-Tuning

O primeiro agente responde perguntas médicas gerais utilizando o modelo Qwen ajustado com o MedQuAD.

Esse agente não consulta os prontuários no ChromaDB. Ele gera respostas com base nos padrões aprendidos durante o treinamento.

As respostas são identificadas como geradas automaticamente e não validadas clinicamente.

### 5.2. Agente RAG

O segundo agente responde perguntas relacionadas aos prontuários sintéticos.

Os registros são armazenados no ChromaDB utilizando embeddings produzidos pelo modelo `paraphrase-multilingual-MiniLM-L12-v2`.

A integração com LangChain permite recuperar documentos por identificador e realizar busca semântica nas observações clínicas.

Para perguntas sobre exames, o assistente utiliza os registros estruturados recuperados. Para perguntas sobre anotações clínicas, o contexto recuperado é enviado ao Qwen original para elaboração de uma resposta.

### 5.3. Identificação dos pacientes

O usuário pode consultar um paciente pelo identificador ou pelo nome completo.

Quando o nome é utilizado, o sistema tenta identificar o registro correspondente e resolver seu ID interno.

Em situações de nomes duplicados, o assistente solicita informações adicionais, como idade ou identificador do prontuário, evitando selecionar arbitrariamente um paciente.

### 5.4. Orquestração com LangGraph

O LangGraph seleciona o caminho da solicitação:

- **RAG:** perguntas relacionadas a pacientes identificados e autorizados.
- **Fine-Tuning:** perguntas médicas gerais.
- **Bloqueio:** solicitações proibidas, acessos indevidos ou identificação insuficiente.

Após a execução, o fluxo passa por um nó de auditoria, responsável por registrar o caminho percorrido.

## 6. Segurança e auditoria

Foram implementados mecanismos demonstrativos para restringir consultas e registrar atividades.

O sistema verifica a autorização fictícia do profissional antes de consultar o prontuário no ChromaDB.

Também foram implementadas regras para bloquear pedidos diretos de prescrição, dosagem ou diagnóstico autônomo.

Os logs registram informações técnicas, incluindo identificador da requisição, data, rota selecionada, status e nós executados, sem armazenar o conteúdo das perguntas ou respostas clínicas.

Esses controles são simplificados. Não representam autenticação hospitalar real, validação profissional nem garantias completas de proteção contra solicitações inadequadas.

## 7. Avaliação funcional do assistente

A integração foi avaliada por meio de testes funcionais.

| Cenário                             | Resultado               |
| ------------------------------------ | ----------------------- |
| Consulta de exames pelo ID           | Aprovado                |
| Consulta de exames pelo nome e idade | Aprovado                |
| Identificação de nome duplicado    | Bloqueio correto        |
| Paciente não identificado           | Bloqueio correto        |
| Acesso não autorizado               | Bloqueio correto        |
| Solicitação direta de prescrição | Bloqueio correto        |
| Execução do agente Fine-Tuning     | Aprovado funcionalmente |

A auditoria confirmou os sete registros correspondentes à validação.

O agente RAG recuperou corretamente o hemograma pendente do paciente fictício `PAC-DEMO-001` e identificou sintomas de cefaleia e tontura presentes nas observações clínicas.

O agente Fine-Tuning também foi executado com seus pesos ajustados, e o LangGraph registrou a rota correspondente.

Esses resultados demonstram o funcionamento dos fluxos avaliados, mas não permitem estimar a acurácia do modelo em situações clínicas reais.

## 8. Análise crítica das respostas

Durante os testes do modelo ajustado, foi observada a capacidade de produzir respostas médicas em linguagem natural, especialmente em inglês, idioma predominante nos exemplos utilizados no treinamento.

Entretanto, em uma resposta sobre diabetes mellitus, o modelo apresentou uma associação conceitualmente incorreta entre diabetes e síndrome metabólica.

Também foram observadas respostas interrompidas pelo limite de tokens configurado para inferência.

Esses resultados indicam que a execução bem-sucedida do Fine-Tuning não garante precisão factual ou completude das respostas.

O protótipo precisa de avaliação adicional com perguntas de teste independentes, critérios objetivos de qualidade e revisão de profissionais habilitados antes de qualquer aplicação médica real.

## 9. Limitações do projeto

As principais limitações identificadas foram:

- O Fine-Tuning utilizou o dataset público MedQuAD, e não documentos hospitalares internos.
- Os prontuários são inteiramente sintéticos.
- As permissões de acesso são simuladas.
- O reconhecimento de intenções e os guardrails utilizam regras baseadas em padrões textuais.
- O modelo pode produzir informações incorretas, imprecisas ou incompletas.
- Não foi realizada validação clínica sistemática.
- A execução exige configurar dependências, reconstruir o índice vetorial e disponibilizar os pesos do modelo ajustado.

Essas limitações devem ser consideradas na interpretação dos resultados.

## 10. Conclusão

O Tech Challenge demonstrou a implementação de um protótipo de assistente médico que integra Full Fine-Tuning, RAG e LangGraph.

O modelo Qwen2.5-0.5B-Instruct foi ajustado com o MedQuAD e posteriormente integrado a uma rota de perguntas médicas gerais.

Paralelamente, foi desenvolvido um agente RAG capaz de consultar informações de 1.000 pacientes sintéticos armazenados no ChromaDB.

Os testes confirmaram o roteamento funcional, a recuperação de informações, o tratamento de nomes ambíguos, os bloqueios demonstrativos e o registro de auditoria.

Como evolução futura, seriam necessárias validações clínicas abrangentes, autenticação real, controles de segurança mais robustos e estudos adicionais sobre a qualidade das respostas geradas.

O projeto atende ao objetivo de demonstrar tecnicamente a integração dessas tecnologias em um ambiente acadêmico controlado.
