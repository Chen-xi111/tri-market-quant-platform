# Publishing to GitHub

Use this checklist for the first public push. The repository already uses Git and its default branch is `main`.

## 1. Decide identity and licensing

1. In GitHub, open **Settings → Emails** and copy either a verified email address or the GitHub-provided `noreply` address.
2. Choose an open-source license deliberately. A public repository without a license remains visible, but normal copyright restrictions still apply.
3. Configure the identity only for this repository:

   ```bash
   git config user.name "YOUR NAME"
   git config user.email "YOUR_GITHUB_OR_NOREPLY_EMAIL"
   git config --get user.name
   git config --get user.email
   ```

Before the first public push, inspect the existing author metadata:

```bash
git log --format='%h %an <%ae>'
```

If earlier commits contain an address you do not want to publish, stop and rewrite that local-only history before adding a GitHub remote. History rewriting changes commit IDs and should be backed up first.

## 2. Run the release gates

```bash
./test_all.sh
.venv/bin/python scripts/release_check.py
git diff --check
git status --short
```

Confirm that only intentional source and documentation changes are listed. Never add `.env`, `secrets.toml`, broker logs, provider caches or account state.

## 3. Commit the release

```bash
git add CHANGELOG.md README.md app_pages pyproject.toml uv.lock systems docs scripts
git status --short
git commit -m "Add cross-market trade decision gates"
```

Run the release gates again after committing. `git status --short` should then be empty.

## 4. Create the GitHub repository

1. On GitHub, select **New repository**.
2. Use a concise name such as `tri-market-quant-platform`.
3. Select **Public**.
4. Do **not** initialize the repository with a README, `.gitignore` or license because those choices must be reconciled with the existing local history.
5. Create the repository and copy its HTTPS URL.

## 5. Add the remote and push

Replace `YOUR_USERNAME` with the GitHub account name:

```bash
git remote add origin https://github.com/YOUR_USERNAME/tri-market-quant-platform.git
git remote -v
git push -u origin main
```

Use a GitHub-supported authentication method if prompted; do not paste credentials into project files.

## 6. Tag and package v0.4.0

Build the archive only after the release commit exists:

```bash
mkdir -p dist
git archive --format=zip --prefix=tri-market-quant-platform-v0.4.0/ --output=dist/tri-market-quant-platform-v0.4.0.zip HEAD
shasum -a 256 dist/tri-market-quant-platform-v0.4.0.zip > dist/tri-market-quant-platform-v0.4.0.zip.sha256
unzip -tq dist/tri-market-quant-platform-v0.4.0.zip
shasum -a 256 -c dist/tri-market-quant-platform-v0.4.0.zip.sha256
git tag -a v0.4.0 -m "v0.4.0"
git push origin v0.4.0
```

On the GitHub repository, open **Releases → Draft a new release**, choose tag `v0.4.0`, generate or write release notes, attach the ZIP and checksum, then publish.

## 7. Verify the public result

- Confirm the README, architecture diagram and changelog render correctly.
- Confirm GitHub Actions passes on `main`.
- Confirm no secrets, cache files or broker/account state appear in the file list or commit history.
- Enable private vulnerability reporting under **Settings → Security**.
- Add a hosted application screenshot only after the public deployment URL is stable.
