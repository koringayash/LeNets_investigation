"""
utils/system_info.py
--------------------
Prints a summary of the current hardware and software environment.

Call this once at the very start of a run so that log files contain
enough context for debugging later — especially useful on cloud VMs
where you might forget which instance type or driver version was used.

Usage
-----
>>> from utils.system_info import SystemInfo
>>> SystemInfo.print(logger=my_logger)
"""

import platform
import logging


class SystemInfo:
    """
    Collects and prints system environment information.

    All methods are static — you never need to instantiate this class.
    Just call SystemInfo.print() directly.

    Example
    -------
    >>> SystemInfo.print()
    ============================================================
      SYSTEM INFORMATION
    ============================================================
      OS          : Linux 5.15.0
      Python      : 3.10.12
      PyTorch     : 2.1.0
      Device      : CUDA 11.8 | GPU: RTX 3080 | VRAM: 10.0 GB
    ============================================================
    """

    @staticmethod
    def print(logger: logging.Logger = None) -> None:
        """
        Collect and display system information.

        Parameters
        ----------
        logger : logging.Logger, optional
            Where to write the info. Falls back to print() if None.
        """
        try:
            import torch
            torch_version = torch.__version__
            if torch.cuda.is_available():
                device_info = (
                    f"CUDA {torch.version.cuda} | "
                    f"GPU: {torch.cuda.get_device_name(0)} | "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )
            else:
                device_info = "CPU only (no CUDA GPU detected)"
        except ImportError:
            torch_version = "not installed"
            device_info   = "unknown"

        lines = [
            "=" * 60,
            "  SYSTEM INFORMATION",
            "=" * 60,
            f"  OS          : {platform.system()} {platform.release()}",
            f"  Python      : {platform.python_version()}",
            f"  PyTorch     : {torch_version}",
            f"  Device      : {device_info}",
            "=" * 60,
        ]
        output = "\n".join(lines)
        if logger:
            logger.info("\n" + output)
        else:
            print(output)