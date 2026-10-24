import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECRETS_FILE = PROJECT_ROOT / ".env"
_SECRET_NAMES = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"}
_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_local_secrets(path: str | Path = DEFAULT_SECRETS_FILE) -> None:
    """Load API keys from a local env file without overriding environment values."""
    secrets_path = Path(path)
    if not secrets_path.is_absolute():
        secrets_path = PROJECT_ROOT / secrets_path
    if not secrets_path.exists():
        return
    if secrets_path.is_symlink():
        raise ValueError(f"Secrets file must not be a symbolic link: {secrets_path}")

    for line_number, raw_line in enumerate(
        secrets_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not _NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid entry in secrets file at line {line_number}.")
        if name not in _SECRET_NAMES:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", maxsplit=1)[0].rstrip()
        if value and not os.environ.get(name):
            os.environ[name] = value
