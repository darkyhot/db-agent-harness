from psycopg2 import OperationalError

from core.exceptions import KERBEROS_USER_MESSAGE, KerberosAuthError, is_kerberos_auth_error


def test_is_kerberos_auth_error_detects_gss_failure():
    exc = OperationalError("FATAL: GSS authentication failed for user")

    assert is_kerberos_auth_error(exc) is True


def test_is_kerberos_auth_error_ignores_regular_db_failure():
    exc = OperationalError("relation does not exist")

    assert is_kerberos_auth_error(exc) is False


def test_is_kerberos_auth_error_checks_exception_chain():
    try:
        try:
            raise OperationalError("Ticket expired in credentials cache")
        except OperationalError as exc:
            raise RuntimeError("Ошибка выполнения запроса") from exc
    except RuntimeError as exc:
        wrapped = exc

    assert is_kerberos_auth_error(wrapped) is True
    assert not isinstance(wrapped, KerberosAuthError)


def test_kerberos_user_message_uses_plain_kinit_instruction():
    assert "kinit в терминале" in KERBEROS_USER_MESSAGE
    assert "`kinit`" not in KERBEROS_USER_MESSAGE
    assert "<REALM>" not in KERBEROS_USER_MESSAGE
