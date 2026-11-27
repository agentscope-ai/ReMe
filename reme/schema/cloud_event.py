"""CloudEvents 1.0 envelopes carried as JSON in an SSE data field."""

import base64
import binascii
from datetime import timezone
import json
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, StrictStr, field_validator, model_validator


class CloudEvent(BaseModel):
    """Validate standard context attributes while retaining payloads and extensions."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)

    specversion: Literal["1.0"]
    id: StrictStr = Field(min_length=1)
    source: StrictStr = Field(min_length=1)
    type: StrictStr = Field(min_length=1)
    time: AwareDatetime | None = None
    subject: StrictStr | None = None
    datacontenttype: StrictStr | None = None
    dataschema: StrictStr | None = None
    data: JsonValue = None
    data_base64: StrictStr | None = None

    @field_validator("time", mode="before")
    @classmethod
    def validate_timestamp(cls, value):
        """JSON CloudEvents timestamps use strings, never Unix epoch numbers."""
        pattern = (
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}" + r"(?:\.[0-9]+)?(?:[Zz]|[+-][0-9]{2}:[0-9]{2})"
        )
        if value is not None and (not isinstance(value, str) or not re.fullmatch(pattern, value)):
            raise ValueError("CloudEvents time must be an RFC 3339 timestamp string")
        return value

    @model_validator(mode="after")
    def validate_context(self):
        """Keep context strings portable and reject invalid JSON envelope combinations."""
        for key in ("id", "source", "type", "subject", "datacontenttype", "dataschema"):
            value = getattr(self, key)
            if value is not None and (not value or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
                raise ValueError(f"Invalid CloudEvents {key}")
        for key in ("source", "dataschema"):
            value = getattr(self, key)
            if value is not None:
                if re.search(r'[^\x21-\x7e]|[<>"{}|\\^`]|%(?![0-9A-Fa-f]{2})', value):
                    raise ValueError(f"Invalid CloudEvents {key} URI")
                parsed = urlsplit(value)
                if key == "dataschema" and not parsed.scheme:
                    raise ValueError("CloudEvents dataschema must be an absolute URI")
        if "data" in self.model_fields_set and self.data_base64 is not None:
            raise ValueError("CloudEvents data and data_base64 are mutually exclusive")
        content_type = (self.datacontenttype or "application/json").partition(";")[0].strip().lower()
        subtype = content_type.partition("/")[2]
        if (
            "data" in self.model_fields_set
            and subtype != "json"
            and not subtype.endswith("+json")
            and not isinstance(self.data, str)
        ):
            raise ValueError("Non-JSON CloudEvents data must be a string; binary payloads use data_base64")
        if self.data_base64 is not None:
            try:
                base64.b64decode(self.data_base64, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Invalid CloudEvents data_base64") from exc
        for key, value in self.__pydantic_extra__.items():
            if not re.fullmatch(r"[a-z0-9]+", key):
                raise ValueError("CloudEvents extension names must use lowercase ASCII letters and digits")
            if value is not None and not (
                isinstance(value, (str, bool)) or type(value) is int and -(2**31) <= value < 2**31
            ):
                raise ValueError("CloudEvents extensions must be strings, booleans or 32-bit integers")
        return self

    def canonical(self):
        """Normalize optional attributes and timestamp offsets while retaining explicit null data."""
        value = {
            key: item
            for key, item in self.model_dump(mode="json", exclude_unset=True).items()
            if item is not None or key == "data"
        }
        if self.time is not None:
            value["time"] = self.time.astimezone(timezone.utc).isoformat()
        return json.dumps(value, sort_keys=True, allow_nan=False)
