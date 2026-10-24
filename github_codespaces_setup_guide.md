# Generative AI Project MVP - Codespaces

This repository contains a modular Python MVP for a generative AI completion
workflow.

## How to Run in GitHub Codespaces

1. **Open in Codespaces:**
   - Push these files to your GitHub repository.
   - Click **Code** -> **Codespaces** -> **Create codespace on main**.

2. **Install Dependencies:**
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

3. **Configure API Key (Optional):**
   Open `.env` in the VS Code Explorer and enter your Gemini key on the line
   `GEMINI_API_KEY=your-new-key`. The app loads this file each time
   it starts. `.env` is ignored by Git; never commit or share it. If the file is
   absent, copy `.env.example` to `.env` first.
   The browser chat and `--ask` do not require a Gemini key. They search the
   official Wikipedia API, update the local index with the matching article,
   and answer with cited excerpts. A Gemini key is only needed for the
   standalone model-generation example.
   To run that example with a local model instead, set `default_provider: local`
   in `config/model_config.yaml`. It uses `google/flan-t5-small`; Transformers
   downloads the model on its first run, so that first run needs network access
   and enough disk space and memory. Afterward, prompts are processed locally.
   To run the local Wikipedia Q&A example, use
   `.venv/bin/python examples/run_local_ai.py`. The Wikipedia search itself
   still requires network access.
   Wikipedia chat responses include an article illustration when the MediaWiki
   API supplies a thumbnail, with a link to its file and license details.
   Successful chat answers also update `data/personal_lexicon.json` with the
   article lead, question context, Wikipedia source, exposure count, and highest
   learning level associated with that concept. This local record persists between
   runs. For generated responses, matching entries can be added to the prompt as
   supplemental context; this does not update model weights or fine-tune the model.
   You can also run the standalone vocabulary example with
   `.venv/bin/python examples/run_vocabulary_growth.py`; it requires access to
   the Wikipedia API.

4. **Run the Application:**
   ```bash
   .venv/bin/python main.py
   ```
   This runs the standalone model-generation example. It uses Gemini when a
   valid `GEMINI_API_KEY` is present; the Wikipedia chat does not need this key.

5. **Run tests (optional):**
   ```bash
   .venv/bin/python -m unittest discover -s tests
   ```

6. **Ingest and query a progressive Wikipedia article (optional):**
   ```bash
   .venv/bin/python main.py \
     --wiki-topic "Inteligência artificial" \
     --ask "O que é inteligência artificial?" \
     --level basico
   ```
   Use `intermediario` or `avancado` to select a different curriculum level.
   The extraction uses the official MediaWiki API, writes an attributed JSON
   snapshot to `data/prompts/`, and indexes each section locally. The question
   is also searched live through Wikipedia; this requires network access.

7. **Build the universal curriculum (optional):**
   ```bash
   .venv/bin/python main.py --universal-curriculum
   ```
   This fetches the curated starter articles from Wikipedia, writes attributed
   JSON files under `data/universal_curriculum/nivel_1_basico/`,
   `nivel_2_intermediario/`, and `nivel_3_avancado/`, and indexes their
   chunks in the configured SQLite vector database. It requires network access
   but does not make Gemini embedding calls. Questions are searched live and
   the matching article is refreshed in the index:
   ```bash
   .venv/bin/python main.py --ask "O que é o método científico?" --level basico
   ```

8. **Open the browser chat interface:**
   ```bash
   .venv/bin/python main.py --chat
   ```
   Open `http://127.0.0.1:8000` locally, or open port 8000 from the Codespaces
   **Ports** panel. Each question is searched live through the official
   Wikipedia API; the found article is saved in the local SQLite index, and the
   answer contains cited excerpts without Gemini generation. The web server
   binds to loopback only and is intended for local development.