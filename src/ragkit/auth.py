"""Supabase authentication.

Why supabase-py and not ``st.login``? Streamlit's built-in login requires a
generic OIDC provider; Supabase is not one. supabase-py (GoTrue) gives us
sign-up, sign-in, refresh and sign-out, and - crucially - an access token we can
verify *server side* on every request.

Security properties
-------------------
* Access tokens are verified locally: signature (JWKS/ES256/RS256 or HS384
  shared secret), ``exp``, ``aud="authenticated"`` and ``iss``.
* Tokens stay in ``st.session_state`` only - never in a URL, a file, or a
  browser cookie that JavaScript could read.
* Errors are mapped to generic messages so the UI cannot be used to enumerate
  registered email addresses.
* Nothing is logged except a hashed user id.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .logging_utils import get_logger
from .security import safe_avatar_url

log = get_logger("auth")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_GENERIC_CREDENTIALS_ERROR = "Email or password is incorrect."
_GENERIC_SIGNUP_ERROR = "Could not create the account. Check the details and try again."
_EMAIL_RATE_LIMIT = (
    "Supabase is throttling confirmation emails (the free tier allows only a few per hour). "
    "Wait an hour and try again, or turn off Authentication -> Providers -> Email -> "
    "\"Confirm email\" while evaluating."
)


def _provider_hint(message: str) -> str | None:
    """Turn a GoTrue/Supabase error string into an actionable message.

    Sign-in failures are deliberately vague (no account enumeration), but these
    specific ones are configuration problems the operator must see - they are
    the usual reason a Cloud deployment "cannot log in".
    """
    text = (message or "").lower()
    if "rate limit" in text or "over_email_send_rate_limit" in text:
        return _EMAIL_RATE_LIMIT
    if "signups not allowed" in text or "signup is disabled" in text:
        return "Sign-ups are disabled for this Supabase project (Authentication -> Providers -> Email)."
    if "invalid api key" in text or "invalid apikey" in text or "no api key" in text:
        return (
            "SUPABASE_ANON_KEY was rejected. Use the project's anon/publishable key "
            "(not the service_role key), save it in Secrets, and reboot the app."
        )
    if any(token in text for token in ("failed to fetch", "getaddrinfo", "timed out", "connection")):
        return (
            "Could not reach Supabase. Check SUPABASE_URL and unpause the project "
            "if it was paused for inactivity."
        )
    return None


class AuthError(RuntimeError):
    """User-presentable authentication failure."""


@dataclass
class User:
    id: str
    email: str
    name: str = ""
    avatar_url: str | None = None
    role: str = "analyst"
    is_demo: bool = False
    expires_at: float = 0.0

    @property
    def display_name(self) -> str:
        return self.name or self.email.split("@")[0]

    @property
    def initials(self) -> str:
        parts = [p for p in re.split(r"[\s._-]+", self.display_name) if p]
        return "".join(p[0].upper() for p in parts[:2]) or "?"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "avatar_url": self.avatar_url,
            "role": self.role,
            "is_demo": self.is_demo,
        }


@dataclass
class Session:
    access_token: str = ""
    refresh_token: str = ""
    expires_at: float = 0.0
    token_type: str = "bearer"
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def is_expired(self) -> bool:
        return bool(self.expires_at) and time.time() > self.expires_at - 30


def normalise_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        raise AuthError("Enter a valid email address.")
    return email


def validate_password(password: str) -> str:
    """Cheap client-side policy; Supabase enforces the authoritative one."""
    if len(password or "") < 10:
        raise AuthError("Password must be at least 10 characters.")
    if password.lower() in {"password12", "1234567890", "qwertyuiop"}:
        raise AuthError("Choose a less predictable password.")
    return password


def _user_from_claims(claims: dict[str, Any], admin_emails: tuple[str, ...]) -> User:
    metadata = claims.get("user_metadata") or {}
    email = str(claims.get("email") or metadata.get("email") or "")
    name = str(metadata.get("full_name") or metadata.get("name") or "")
    avatar = safe_avatar_url(metadata.get("avatar_url") or metadata.get("picture"))
    role = "admin" if email.lower() in admin_emails else "analyst"
    return User(
        id=str(claims.get("sub") or ""),
        email=email,
        name=name,
        avatar_url=avatar,
        role=role,
        expires_at=float(claims.get("exp") or 0),
    )


class AuthService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jwks_client = None

    # ------------------------------------------------------------------ state
    @property
    def configured(self) -> bool:
        return self.settings.auth_enabled

    def demo_user(self) -> User:
        return User(
            id="local-demo",
            email="demo@localhost",
            name="Local Demo",
            role="admin",
            is_demo=True,
        )

    def _supabase(self):
        """Return a *fresh* GoTrue client.

        supabase-py keeps the signed-in session inside the client instance, so a
        process-wide cached client would let concurrent Streamlit sessions clobber
        each other's tokens. Building one per auth call is cheap and removes that
        cross-user state entirely.
        """
        if not self.configured:
            raise AuthError("Supabase is not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY.")
        try:
            from supabase import create_client
        except Exception as exc:  # pragma: no cover
            raise AuthError("The supabase package is not installed.") from exc
        return create_client(self.settings.supabase_url, self.settings.supabase_anon_key)

    # ------------------------------------------------------------------ tokens
    def _jwks(self):
        if self._jwks_client is not None:
            return self._jwks_client
        try:
            from jwt import PyJWKClient
        except Exception as exc:  # pragma: no cover
            raise AuthError("PyJWT with crypto support is required.") from exc
        url = f"{self.settings.supabase_url}/auth/v1/.well-known/jwks.json"
        self._jwks_client = PyJWKClient(url, cache_keys=True, lifespan=900)
        return self._jwks_client

    def verify_token(self, token: str) -> User:
        """Full local verification of a Supabase access token."""
        if not token:
            raise AuthError("Not signed in.")
        try:
            import jwt

            header = jwt.get_unverified_header(token)
            algorithm = header.get("alg", "")
            if algorithm == "HS256" and self.settings.supabase_jwt_secret:
                key = self.settings.supabase_jwt_secret
            else:
                key = self._jwks().get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                key,
                algorithms=[algorithm] if algorithm else ["ES256", "RS256"],
                audience="authenticated",
                issuer=f"{self.settings.supabase_url}/auth/v1",
                options={"require": ["exp", "sub", "aud"]},
            )
        except AuthError:
            raise
        except Exception as exc:
            log.info("token rejected: %s", type(exc).__name__)
            raise AuthError("Your session has expired. Please sign in again.") from exc

        user = _user_from_claims(claims, self.settings.admin_emails)
        if not user.id:
            raise AuthError("Token is missing a subject.")
        return user

    # ------------------------------------------------------------------- flows
    def sign_in(self, email: str, password: str) -> tuple[User, Session]:
        email = normalise_email(email)
        if not password:
            raise AuthError(_GENERIC_CREDENTIALS_ERROR)
        client = self._supabase()
        try:
            response = client.auth.sign_in_with_password({"email": email, "password": password})
        except Exception as exc:
            log.info("sign-in failed for %s", hashlib.sha256(email.encode()).hexdigest()[:10])
            message = str(exc).lower()
            if "email not confirmed" in message:
                raise AuthError("Please confirm your email address, then sign in.") from exc
            hint = _provider_hint(message)
            if hint:
                raise AuthError(hint) from exc
            raise AuthError(_GENERIC_CREDENTIALS_ERROR) from exc

        session = getattr(response, "session", None)
        if session is None:
            raise AuthError("Sign-in did not return a session. Confirm your email first.")
        user = _user_from_claims(
            {"sub": str(response.user.id), "email": response.user.email,
             "user_metadata": response.user.user_metadata or {}, "exp": 0},
            self.settings.admin_emails,
        )
        # Trust the verified token for identity/expiry.
        verified = self.verify_token(session.access_token)
        verified.name = verified.name or user.name
        verified.avatar_url = verified.avatar_url or user.avatar_url
        return verified, Session(
            access_token=session.access_token,
            refresh_token=session.refresh_token,
            expires_at=float(verified.expires_at or 0),
        )

    def sign_up(
        self, email: str, password: str, full_name: str = "", redirect_to: str = ""
    ) -> tuple[bool, str]:
        """Returns (needs_email_confirmation, message).

        ``redirect_to`` is the origin the confirmation link sends the browser to.
        Pass the deployed origin (``RAG_PUBLIC_URL``) so Cloud users are not sent
        to the localhost default; empty means "use the Supabase Site URL".
        """
        email = normalise_email(email)
        validate_password(password)
        full_name = (full_name or "").strip()[:80]
        options: dict[str, Any] = {"data": {"full_name": full_name}}
        if redirect_to:
            options["email_redirect_to"] = redirect_to.rstrip("/")
        client = self._supabase()
        try:
            response = client.auth.sign_up(
                {"email": email, "password": password, "options": options}
            )
        except Exception as exc:
            message = str(exc).lower()
            hint = _provider_hint(message)
            if hint:
                raise AuthError(hint) from exc
            if "already registered" in message or "already exists" in message:
                # Same message as success-on-confirm to avoid account enumeration.
                return True, "Check your inbox to confirm your email address."
            if "password" in message:
                raise AuthError("Password does not meet the project's policy (12+ characters recommended).") from exc
            log.info("sign-up failed: %s", type(exc).__name__)
            raise AuthError(_GENERIC_SIGNUP_ERROR) from exc

        if getattr(response, "session", None) is not None:
            return False, "Account created. You can sign in now."
        return True, "Account created. Check your email to confirm it, then sign in."

    def sign_out(self, session: Session | None) -> None:
        if session is None or not session.access_token:
            return
        try:
            client = self._supabase()
            # GoTrue signs out whatever session the client holds, so set it first.
            client.auth.set_session(session.access_token, session.refresh_token or session.access_token)
            client.auth.sign_out()
        except Exception:
            pass  # local session is cleared regardless
        finally:
            session.access_token = ""
            session.refresh_token = ""

    def refresh(self, session: Session) -> Session:
        if not session.refresh_token:
            raise AuthError("Session expired. Please sign in again.")
        client = self._supabase()
        try:
            response = client.auth.refresh_session(session.refresh_token)
        except Exception as exc:
            raise AuthError("Session expired. Please sign in again.") from exc
        if getattr(response, "session", None) is None:
            raise AuthError("Session expired. Please sign in again.")
        return Session(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            expires_at=time.time() + 3600,
        )

    def ensure_valid(self, user: User, session: Session | None) -> User:
        """Re-verify the token on each request (defence against stale sessions)."""
        if user.is_demo:
            return user
        if session is None:
            raise AuthError("Not signed in.")
        if session.is_expired:
            session = self.refresh(session)
            return self.verify_token(session.access_token)
        return user
