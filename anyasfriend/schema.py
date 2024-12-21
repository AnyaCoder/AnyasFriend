from datetime import datetime
from enum import Enum
from typing import AnyStr, Dict, List, Literal
from uuid import UUID

from pydantic import BaseModel


class AnyaData(BaseModel):

    class Type(Enum):
        AUDIO = "audio"
        EVENT = "event"
        TEXT = "text"

    class Event(Enum):
        ACCEPT_TEXT = "accept_text"
        REJECT_TEXT = "reject_text"
        ACCEPT_AUDIO = "accept_audio"
        REJECT_AUDIO = "reject_audio"
        CANCEL = "cancel"

    class Content(BaseModel):
        messages: List[Dict[str, AnyStr]]
        who: str

    dtype: Type
    content: str | bytes | dict | Content
    identifier: UUID  # uuid4
    timestamp: datetime
