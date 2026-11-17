import hashlib
import hmac
import secrets

from config import PBKDF2_ITERATIONS, SALT_BYTES, SESSION_TOKEN_BYTES


def _derive(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )


def hash_password(password: str) -> str:
    """Return ``salt_hex:hash_hex`` for the given password."""
    salt = secrets.token_bytes(SALT_BYTES)
    return f"{salt.hex()}:{_derive(password, salt).hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, hash_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(_derive(password, salt), expected)


def generate_session_token() -> str:
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def tokens_match(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
