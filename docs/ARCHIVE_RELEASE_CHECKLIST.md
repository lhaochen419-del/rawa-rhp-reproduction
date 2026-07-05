# Archive Release Checklist

- Create a GitHub repository from this package.
- Confirm final software/data licence with all authors.
- Remove private paths from logs and manuscripts.
- Run `python scripts/verify_reproduction.py --package-root .`.
- Run `python scripts/make_tables_figures.py --package-root .`.
- Create a GitHub release and archive it with Zenodo.
- Replace `[private reviewer link]` and `[DOI after public release]` in the manuscript.
