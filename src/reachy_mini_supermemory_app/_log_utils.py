"""Shared logging helpers for the supermemory app.

The startup path runs before the conversation app configures root logging,
so plain ``logger.info(...)`` lines get dropped. Several modules independently
re-implemented the same fix: print to stderr AND emit through a logger.
This module owns that pattern so a future change to logging policy (file
sink, structured logs, journald metadata) happens in one place.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def startup_log(message: str, *, logger: Optional[logging.Logger] = None) -> None:
    """Emit a startup-time message through both stderr and a logger.

    stderr is the bootstrap channel — guaranteed to reach the systemd journal
    even before the conversation app configures root logging. The logger
    mirror ensures the same line lands in any structured-log handler that
    gets wired up later in the boot sequence.

    Pass the caller's module logger to keep ``record.name`` accurate;
    falls back to this module's logger if omitted.
    """
    print(message, file=sys.stderr, flush=True)
    (logger or _default_logger).info(message)


_default_logger = logging.getLogger(__name__)
