from __future__ import annotations

from typing import Literal

import pygame

from .base import BaseHUD
from .fuzzy import FuzzyHUD
from .transformer import TransformerHUD


def create_hud(
    agent_mode: Literal["transformer", "fuzzy"],
    surface: pygame.Surface,
    panel_rect: pygame.Rect,
) -> BaseHUD:
    if agent_mode == "transformer":
        return TransformerHUD(surface, panel_rect)
    return FuzzyHUD(surface, panel_rect)


__all__ = [
    "BaseHUD",
    "FuzzyHUD",
    "TransformerHUD",
    "create_hud",
]
