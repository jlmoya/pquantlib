# Releasing `pquantlib`

How to publish the **`pquantlib`** package (sources in [`pquantlib/`](pquantlib/)) to PyPI.
The other workspace members (`pquantlib-contrib`, `pquantlib-helpers`, `pquantlib-samples`,
`pquantlib-showcase`) are **not** published by this process.

- **Versioning** tracks the ported C++ QuantLib release (currently `1.42.1`). The git tag is `v<version>`.
- **Publishing is automated** by [`.github/workflows/release.yml`](.github/workflows/release.yml) —
  GitHub Actions builds `python -m build pquantlib` → `pquantlib/dist/` and uploads to PyPI/TestPyPI
  via **OIDC Trusted Publishing** (no API tokens).

## One-time setup — Trusted Publishers

Configure a Trusted Publisher on each index so GitHub can publish without a stored token. Because the
project doesn't exist on either index yet, add a **pending** publisher (the first publish creates it).

**TestPyPI** — <https://test.pypi.org> → *Account settings → Publishing → Add a pending publisher*:

| Field | Value |
|---|---|
| PyPI Project Name | `pquantlib` |
| Owner | `jlmoya` |
| Repository name | `pquantlib` |
| Workflow filename | `release.yml` |
| Environment name | `testpypi` |

**PyPI** — <https://pypi.org> → same form, but **Environment name = `pypi`**.

The GitHub environments `testpypi` / `pypi` are created automatically on first run. Add required
reviewers under *Settings → Environments* if you want a manual approval gate before a publish.

## Cutting a release

1. **Bump the version** in [`pquantlib/pyproject.toml`](pquantlib/pyproject.toml) (`version = "X.Y.Z"`)
   and sync the lockfile:
   ```bash
   uv lock
   git add pquantlib/pyproject.toml uv.lock
   git commit -s -m "release: pquantlib X.Y.Z"
   git push origin main
   ```

2. **Dry-run on TestPyPI first** — GitHub → *Actions → release → Run workflow* → `target: testpypi`.
   Then sanity-check it (Python ≥ 3.14; the extra index supplies numpy/scipy/mpmath, which aren't on TestPyPI):
   ```bash
   uv pip install --index-url https://test.pypi.org/simple/ \
                  --extra-index-url https://pypi.org/simple/ pquantlib
   ```
   Confirm <https://test.pypi.org/project/pquantlib/> renders the README, version, and license.

3. **Publish to PyPI** — tag and push:
   ```bash
   git tag -a vX.Y.Z -m "PQuantLib vX.Y.Z"
   git push origin vX.Y.Z          # the pushed v* tag triggers the publish-pypi job
   ```
   Confirm <https://pypi.org/project/pquantlib/>.

## Notes

- The workflow builds **only** the `pquantlib` package (`python -m build pquantlib`), not the workspace root.
- `skip-existing: true` makes re-running a publish for an already-uploaded version a safe no-op — but a
  new upload always needs a **new** version; PyPI/TestPyPI never allow overwriting an existing version.
- The wheel is pure-Python (`py3-none-any`), requires **Python ≥ 3.14**, and ships `py.typed` + the
  `BSD-3-Clause` license (`dist-info/licenses/LICENSE.TXT`).
- **Manual fallback** (no CI): `uv build --package pquantlib` then
  `uv publish --index testpypi pquantlib/dist/*` (or `twine upload`).
