# avionmesh

Python library for Avi-on Bluetooth mesh lights (CSRMesh protocol).

## Release Process

Releases are fully automated via GitHub Actions. Pushing a version tag triggers:

1. Build wheel and source tarball
2. Publish to PyPI (trusted publishing)
3. Create GitHub release with attached artifacts

### Steps

1. Bump version in `pyproject.toml`
2. Commit: `git commit -am "Bump to v0.X.0"`
3. Tag and push: `git tag v0.X.0 && git push && git push --tags`

The workflow `.github/workflows/workflow.yml` runs on `push: tags: ['v*']`.

## Architecture

- Uses `recsrmesh` for CSRMesh protocol (association, mesh commands)
- MQTT bridge for Home Assistant integration
- Independent MQTT and mesh operation

## Code Style

- Ruff for linting/formatting (`ruff check --fix src/`)
- mypy for type checking (`mypy src/`)
