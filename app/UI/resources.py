from __future__ import annotations

import sys
from pathlib import Path


def resource_path(*parts: str) -> str:
    """
    Абсолютный путь к ресурсу.

    - Dev: относительно корня репозитория
    - PyInstaller: относительно sys._MEIPASS (куда распаковывается --onefile)
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return str(Path(base, *parts))

    repo_root = Path(__file__).resolve().parents[2]  # .../app/UI -> .../<root>
    return str(repo_root.joinpath(*parts))
