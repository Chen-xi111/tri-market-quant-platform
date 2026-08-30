# Security policy

## Supported version

Security fixes are applied to the latest release on the `main` branch.

## Reporting a vulnerability

Use the repository host's private vulnerability-reporting feature when it is available. Do not include credentials, account identifiers, broker logs or exploit details in a public issue. If private reporting is unavailable, open a minimal issue requesting a secure contact channel without disclosing the vulnerability.

## Safety scope

- The public portfolio is read-only and does not expose broker controls.
- Local execution adapters are intended for paper accounts by default.
- Credentials belong in ignored local environment or Streamlit secret files.
- Market-data and broker behavior can change independently of this repository; validate external configuration before use.
