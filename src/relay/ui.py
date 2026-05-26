"""Terminal styling for the Relay CLI.

ANSI escape codes only — no extra dependency. The CLI uses these for the
welcome banner and to colour status output across ``relay init / start /
stop / status``. Colour is suppressed when stdout is not a TTY, when
``NO_COLOR`` is set, or when ``RELAY_NO_COLOR=1``.
"""

from __future__ import annotations

import os
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Amber/yellow accent (matches the dashboard theme), plus the usual semantic colours.
AMBER = "\033[38;5;214m"
GREEN = "\033[38;5;42m"
RED = "\033[38;5;203m"
YELLOW = "\033[38;5;221m"
GREY = "\033[38;5;245m"
DIM_GREY = "\033[38;5;240m"


def color_enabled() -> bool:
    """Return whether the terminal should receive escape codes."""
    if os.getenv("NO_COLOR") is not None:
        return False
    if os.getenv("RELAY_NO_COLOR") == "1":
        return False
    return sys.stdout.isatty()


def style(text: str, *codes: str) -> str:
    """Wrap ``text`` in the given ANSI codes when colour is enabled."""
    if not color_enabled() or not codes:
        return text
    return "".join(codes) + text + RESET


def banner(subtitle: str = "distributed llm serving for consumer machines") -> str:
    """Return the ASCII Relay banner with the configured colour."""
    art = (
        " ____  _____ _        _ __   __\n"
        "|  _ \\| ____| |      / \\\\ \\ / /\n"
        "| |_) |  _| | |     / _ \\\\ V / \n"
        "|  _ <| |___| |___ / ___ \\| |  \n"
        "|_| \\_\\_____|_____/_/   \\_\\_|  \n"
    )
    return (
        "\n"
        + style(art, AMBER, BOLD)
        + style(f"  {subtitle}\n", DIM_GREY)
        + style("  https://github.com/Mati02K/relay\n", DIM_GREY)
        + "\n"
    )


def ok(text: str) -> str:
    """Format a success label."""
    return style(text, GREEN, BOLD)


def warn(text: str) -> str:
    """Format a warning label."""
    return style(text, YELLOW, BOLD)


def err(text: str) -> str:
    """Format an error label."""
    return style(text, RED, BOLD)


def accent(text: str) -> str:
    """Format an accent label (used for the brand colour)."""
    return style(text, AMBER, BOLD)


def muted(text: str) -> str:
    """Format dim secondary text."""
    return style(text, DIM_GREY)


def status_label(running: bool, detail: str) -> str:
    """Pick the right colour for a process status word.

    ``detail`` strings come from ``ProcessStatus.detail`` (see
    :mod:`relay.supervisor`) and may contain things like
    ``"already running"`` or ``"stopped"``.
    """
    if running:
        return ok("running")
    text = detail.lower()
    if "already" in text:
        return warn("already")
    if "stop" in text:
        return muted("stopped")
    if "no pid" in text:
        return muted("inactive")
    return muted(detail)
