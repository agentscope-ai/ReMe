"""Atomic instance settings with credentials encrypted at rest."""

import json
import os
import sys
from pathlib import Path
from typing import Literal
from uuid import uuid4

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field

from .config import Profile, SECRET_FIELDS, Settings


class StoredProfile(BaseModel):
    """Validate the envelope before looking up or decrypting its credentials."""

    revision: int = Field(ge=0)
    key_id: str | None
    settings: dict


class OperationRecord(BaseModel):
    """One operation in the persisted history."""

    id: str
    phase: Literal["accepted", "restarting", "succeeded", "failed"]
    error: str | None


class OperationHistory(BaseModel):
    """Validate the operation log independently of the active profile."""

    operations: dict[str, OperationRecord]


def write_private(path: Path, content: str) -> None:
    """Publish a private file atomically after flushing its contents."""
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class SettingsStore:
    """Keep ordinary values readable and encrypt API keys with a separate master key."""

    def __init__(self, directory: Path):
        self.path = directory / "settings.json"
        self.key_id: str | None = None
        self._cipher: Fernet | None = None

    def _fernet(self, create: bool) -> Fernet:
        if self._cipher is not None:
            return self._cipher
        key = os.getenv("REME_ENCRYPTION_KEY")
        if not key:
            try:
                if sys.platform == "darwin":
                    from keyring.backends.macOS import Keyring

                    backend = Keyring()
                elif sys.platform == "win32":
                    from keyring.backends.Windows import WinVaultKeyring

                    backend = WinVaultKeyring()
                else:
                    from keyring.backends.SecretService import Keyring

                    backend = Keyring()
                if self.key_id is None:
                    self.key_id = uuid4().hex
                key = backend.get_password("ReMe", self.key_id)
                if not key and create:
                    key = Fernet.generate_key().decode("ascii")
                    backend.set_password("ReMe", self.key_id, key)
            except Exception as error:  # pylint: disable=broad-except
                raise ValueError(
                    "Cannot access the system credential store. Server deployments can set REME_ENCRYPTION_KEY.",
                ) from error
        if not key:
            raise ValueError("The saved settings encryption key is unavailable")
        try:
            self._cipher = Fernet(key.encode("ascii"))
        except (ValueError, UnicodeError) as error:
            raise ValueError("REME_ENCRYPTION_KEY must be a Fernet key") from error
        return self._cipher

    def load(self) -> Profile | None:
        """Read one committed generation without exposing decrypted keys in errors."""
        if not self.path.exists():
            return None
        try:
            payload = StoredProfile.model_validate_json(self.path.read_text(encoding="utf-8"))
            self.key_id = payload.key_id
            values = payload.settings
            for field in SECRET_FIELDS:
                encrypted = values.get(field)
                if encrypted is not None and not isinstance(encrypted, str):
                    raise ValueError("Invalid encrypted credential")
                if encrypted:
                    values[field] = self._fernet(create=False).decrypt(encrypted.encode("ascii")).decode("utf-8")
            return Profile(revision=payload.revision, settings=Settings.model_validate(values))
        except (OSError, ValueError, InvalidToken) as error:
            raise ValueError("Cannot read saved settings. Restore settings.json and its encryption key.") from error

    def prepare(self, profile: Profile) -> str:
        """Encrypt a candidate before stopping the running worker."""
        values = profile.settings.private_values()
        for field in SECRET_FIELDS:
            if values[field]:
                values[field] = self._fernet(create=True).encrypt(values[field].encode("utf-8")).decode("ascii")
        return (
            json.dumps(
                {"revision": profile.revision, "key_id": self.key_id, "settings": values},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )

    def commit(self, prepared: str) -> None:
        """Replace the saved generation only after the new worker is ready."""
        write_private(self.path, prepared)
