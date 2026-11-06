# conftest.py —— 保证 pytest 运行时仓库根目录在 sys.path
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
