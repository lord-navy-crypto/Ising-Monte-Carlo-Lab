# Release checklist

- [ ] Update `APP_VERSION` in `app.py` and `version` in `pyproject.toml`.
- [ ] Update `CHANGELOG.md`.
- [ ] Run `python -m compileall -q app.py ising_lab tools tests`.
- [ ] Run `python -m pytest -q` in an environment with all development dependencies.
- [ ] Check macOS and Windows launcher behavior, including the case where port 8501 is occupied.
- [ ] Confirm numerical reference tests still pass.
- [ ] Confirm plots precede default data summaries and complete data remain opt-in.
- [ ] Inspect `git status` for generated environments, secrets, or local exports.
- [ ] Tag the release only after CI is green.

## Public-release note

No open-source license is selected in this package. Add the license that the repository owner wants before inviting external reuse or redistribution.
