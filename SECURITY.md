# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x.x   | yes       |

## Reporting a Vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Email devneatharva@gmail.com with:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

You will receive a response within 48 hours. Confirmed vulnerabilities will be patched and released promptly.

## Security Considerations

- The API does **not** store or log any personally identifiable information (PII).
- All inputs are validated via Pydantic before reaching the model.
- The bundled `insurance.csv` is anonymised synthetic/public data — no real PII.
- Rate limiting is enforced by `RateLimitMiddleware` (default: 100 req/min per IP).
- Do not expose the API publicly without placing it behind a reverse proxy (nginx/caddy) with TLS.
- Set `APP_ENV=production` in production environments to enable stricter logging and disable debug output.
