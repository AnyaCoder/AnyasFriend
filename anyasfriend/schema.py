from datetime import datetime
from enum import Enum
from typing import Literal
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

    dtype: Type
    content: str | bytes | dict
    identifier: UUID  # uuid4
    timestamp: datetime
