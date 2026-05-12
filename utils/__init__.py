"""
utils/__init__.py
-----------------
Public API for the utils package.

Instead of writing long import paths everywhere:
    from utils.logger import get_logger
    from utils.timer import Timer

Any file in the project can write the shorter form:
    from utils import get_logger, Timer

Beginners: This file tells Python "when someone imports utils,
here is what they can access." It acts like a table of contents
for the whole package.
"""

from utils.seed        import set_seed
from utils.timer       import Timer
from utils.logger      import get_logger
from utils.system_info import SystemInfo
from utils.metrics     import MetricWriter

__all__ = [
    "set_seed",
    "Timer",
    "get_logger",
    "SystemInfo",
    "MetricWriter",
]