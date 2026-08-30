# Public release checklist

Run these steps before pushing the repository or publishing a release archive.

## Automated checks

```bash
python scripts/release_check.py
./test_all.sh
```

The preflight command verifies the project version, required public files, nine fixture checksums, fixture schemas and row counts, tracked private-runtime paths, and the single-port public launcher boundary.

## Repository settings

- Set the intended Git author name and email before the public push.
- Choose and add a license deliberately; no license is selected automatically by this project.
- Enable GitHub Actions and private vulnerability reporting.
- Protect `main` if other contributors will have write access.
- Keep `.env`, `secrets.toml`, provider caches, broker logs and account state untracked.

## Deployment settings

- Set `QUANT_PUBLIC_MODE=true`.
- Expose only port 8500.
- Do not mount local broker credentials or runtime cache directories.
- Confirm `/_stcore/health` returns HTTP 200 after deployment.
- Open the China, Hong Kong, US and methodology pages once on desktop and mobile widths.

## Release artifacts

- Update `CHANGELOG.md` and the version in `pyproject.toml` and `uv.lock`.
- Build the archive from committed files with `git archive`.
- Publish a SHA-256 checksum beside the archive.
- Add real application screenshots after a hosted URL is available.
