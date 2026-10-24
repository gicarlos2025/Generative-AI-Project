# Generative AI Project

MVP modular em Python para experimentar geração de texto, recuperação aumentada por busca (RAG) e aprendizagem assistida por conteúdo da Wikipédia. Inclui um chat local que pesquisa artigos em tempo real, apresenta trechos com fontes e organiza o conteúdo por nível de aprendizagem.

## Funcionalidades

- Pipeline RAG com fragmentação de texto, embeddings e índice vetorial SQLite.
- Provedores de geração configuráveis: Google Gemini, OpenAI, Anthropic e modelo local com Hugging Face.
- Chat web local que pesquisa a Wikipédia em português (ou outro idioma configurado) e responde com trechos atribuídos às fontes.
- Ingestão de artigos e de um currículo universal organizado em níveis básico, intermediário e avançado.
- Memória cognitiva e vocabulário pessoal armazenados localmente para acompanhar conceitos consultados.
- Exemplos executáveis e testes com `unittest`.

## Requisitos

- Python 3.11 ou superior.
- Acesso à internet para consultar a Wikipédia e, na primeira execução do provedor local, baixar o modelo Hugging Face.
- Uma chave de API para gerar respostas com Gemini, OpenAI ou Anthropic. Sem chave, o cliente Gemini retorna uma resposta simulada; o chat da Wikipédia continua a apresentar conteúdo recuperado e citado.

## Instalação

```bash
git clone https://github.com/gicarlos2025/Generative-AI-Project.git
cd Generative-AI-Project
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Configure as chaves de API conforme o provedor escolhido:

```bash
cp .env.example .env
```

Edite `.env` e preencha apenas a chave necessária (`GEMINI_API_KEY`, `OPENAI_API_KEY` ou `ANTHROPIC_API_KEY`). O arquivo `.env` é ignorado pelo Git; não compartilhe nem faça commit dele.

O provedor e os parâmetros dos modelos são configurados em [`config/model_config.yaml`](config/model_config.yaml). O padrão é Gemini. Para escolher outro provedor, altere `default_provider` para `openai`, `anthropic` ou `local`. O provedor local usa `google/flan-t5-small`; o download inicial exige internet, espaço em disco e memória suficientes.

## Como executar

### Geração e RAG

```bash
.venv/bin/python main.py
```

O comando executa o exemplo de geração e consulta RAG. Com uma chave válida, a resposta é gerada pelo provedor selecionado. Sem chave Gemini, a geração retorna uma resposta simulada, não uma resposta produzida pelo modelo.

### Chat local com Wikipédia

```bash
.venv/bin/python main.py --chat
```

Abra [http://127.0.0.1:8000](http://127.0.0.1:8000). No GitHub Codespaces, abra também a porta `8000` pela aba **Ports**. Para usar outra porta:

```bash
.venv/bin/python main.py --chat --port 8080
```

O chat pesquisa a Wikipédia para cada pergunta e mostra trechos do artigo encontrado, com ligação para a fonte. Não depende de uma chave de modelo generativo. O servidor é limitado a loopback e destina-se a desenvolvimento local.

### Pergunta pela linha de comando

```bash
.venv/bin/python main.py --ask "O que é inteligência artificial?" --level basico
```

Os níveis aceitos são `basico`, `intermediario` e `avancado`. A consulta pesquisa a Wikipédia em tempo real e requer acesso à internet.

### Ingerir conteúdo da Wikipédia

Indexe um artigo antes de consultar:

```bash
.venv/bin/python main.py \
  --wiki-topic "Inteligência artificial" \
  --ask "O que é inteligência artificial?" \
  --level basico
```

Crie o currículo universal e indexe seus artigos:

```bash
.venv/bin/python main.py --universal-curriculum
```

Os artigos e o índice são guardados localmente nos diretórios `data/prompts/`, `data/universal_curriculum/` e `data/`, conforme a configuração. Para alterar o idioma da Wikipédia:

```bash
.venv/bin/python main.py --wiki-language en --ask "What is artificial intelligence?"
```

### Exemplos adicionais

```bash
.venv/bin/python examples/run_local_ai.py
.venv/bin/python examples/run_vocabulary_growth.py
.venv/bin/python examples/cognitive_loop.py
```

`run_local_ai.py` e `run_vocabulary_growth.py` consultam a Wikipédia; o primeiro também carrega o modelo local configurado.

## Testes

```bash
.venv/bin/python -m unittest discover -s tests
```

## Estrutura do projeto

```text
config/                 Configurações de modelos e logging
examples/               Exemplos de geração, RAG e aprendizagem
src/cognition/          Memória cognitiva e vocabulário
src/llm/                Clientes dos provedores de modelos
src/prompt_engineering/ Construção e encadeamento de prompts
src/rag/                Fragmentação, embeddings, índice e pipeline RAG
src/utils/              Wikipédia, configuração, cache e utilitários
src/web/                Servidor e interface do chat local
tests/                  Testes automatizados
```

## Dados e atribuição

Os dados gerados durante a execução ficam em `data/`, diretório excluído do Git. O índice vetorial usa SQLite e os registros de vocabulário e memória cognitiva são persistidos localmente.

O conteúdo da Wikipédia está disponível sob a licença [Creative Commons Atribuição-CompartilhaIgual 4.0 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/deed.pt_BR). As respostas incluem a ligação para o artigo de origem; ao reutilizar o conteúdo, preserve a atribuição e cumpra os termos da licença.

## Licença

Este projeto está sob a licença MIT. Consulte [`LICENSE`](LICENSE).
