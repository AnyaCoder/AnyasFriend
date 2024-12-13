# anyasfriend/components/interfaces/core.py

import asyncio
from abc import ABC


class Core(ABC):
    cancel_event = asyncio.Event()
    continue_event = asyncio.Event()
