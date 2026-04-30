"""Application-level exception types and detectors."""

from __future__ import annotations


class DatabaseConnectionError(RuntimeError):
    """Base error for database connection failures."""


class KerberosAuthError(DatabaseConnectionError):
    """Kerberos/GSSAPI authentication failed, likely due to an expired ticket."""


_KERBEROS_MARKERS = (
    "gssapi",
    "kerberos",
    "krb5",
    "ticket expired",
    "credentials cache",
    "no credentials cache",
    "gss authentication",
)

KERBEROS_USER_MESSAGE = (
    "Не могу подключиться к БД: похоже, истёк Kerberos-тикет. "
    "Перевыпустите его: `kinit <ваш_логин>@<REALM>`, затем повторите запрос."
)


def is_kerberos_auth_error(exc: BaseException) -> bool:
    """Return True when an exception chain looks like Kerberos auth failure."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        args = current.args or (str(current),)
        parts.extend(str(arg) for arg in args)
        parts.append(str(current))
        current = current.__cause__ or current.__context__

    text = " ".join(parts).lower()
    return any(marker in text for marker in _KERBEROS_MARKERS)
