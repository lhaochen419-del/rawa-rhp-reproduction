# Archive Release Checklist

- Create a GitHub repository from this package.
- Confirm final software/data licence with all authors.
- Remove private paths from logs and manuscripts.
- Run `python scripts/verify_reproduction.py --package-root .`.
- Run `python scripts/make_tables_figures.py --package-root .`.
- GitHub release archived with Zenodo DOI:
  https://doi.org/10.5281/zenodo.21200028.
  - DOI and repository links replaced in the manuscript and citation metadata.
