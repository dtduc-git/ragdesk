# Access tokens

Access tokens issued by the auth service expire after 60 minutes. Clients must
use the refresh token to renew an expired access token without re-prompting the
user.

Desktop and native apps must use the OAuth 2.0 authorization code flow with
PKCE (RFC 7636) and a loopback redirect (RFC 8252). Never ship a client secret
inside a desktop binary.

## Common failures

- `401 Unauthorized` with `invalid_token`: the access token expired and the
  refresh token was rotated or revoked.
- `403 Forbidden` with `insufficient_scope`: the token is valid but missing the
  required scope. Request the scope again through the authorization flow.

Refresh tokens are single-use: every refresh returns a new refresh token and
invalidates the previous one.
