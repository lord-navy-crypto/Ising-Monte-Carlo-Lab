# Security

This project is a local scientific application. It is not designed to expose a public network service or to handle secrets.

## Reporting a vulnerability

Please report security problems privately to the repository owner rather than posting exploit details in a public issue. Include the affected version, operating system, reproduction steps, and the smallest practical proof of concept.

## Local-use assumptions

- The launchers bind Streamlit to `localhost`.
- `.streamlit/secrets.toml` is excluded from Git.
- Uploaded CSV/JSON data should be treated as untrusted input and validated before adding new parsing features.
- Do not commit credentials, API keys, private datasets, or personal data.
