# Generative AI Project MVP - Codespace Export

This repository provides a modular, production-ready MVP for building Generative AI applications according to best software engineering practices.

## How to Run in GitHub Codespaces

1. **Open in Codespaces:**
   - Push these files to your GitHub repository.
   - Click **Code** -> **Codespaces** -> **Create codespace on main**.

2. **Install Dependencies:**
   Run the following command in the terminal:
   ```bash
   pip install pyyaml openai pydantic
   ```

3. **Configure API Key (Optional):**
   If you have an OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   *Note: If no key is set, the application automatically runs in mock mode.*

4. **Run the Application:**
   ```bash
   python app_mvp.py
   ```