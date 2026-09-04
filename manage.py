#!/usr/bin/env python3
"""Django 管理入口。

用法::

    python manage.py migrate
    python manage.py runserver 7080
"""

import os
import sys
from pathlib import Path


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
    # 仓库根加入 sys.path，让 `agent` 与 `server` 两个顶层包都能直接 import
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
