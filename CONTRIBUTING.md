# Contributing

Thank you for your interest in contributing to Medical Insurance Cost Prediction!

## Getting Started

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/Medical-Insurance-Cost-Prediction
   cd Medical-Insurance-Cost-Prediction
   ```

2. Install development dependencies:
   ```bash
   make install
   ```

3. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Making Changes

- Create a feature branch: `git checkout -b feat/my-feature`
- Keep commits atomic and use [Conventional Commits](https://www.conventionalcommits.org/)
- Run tests before pushing: `make test`
- Run lint before pushing: `make lint`

## Pull Request Guidelines

- Target the `main` branch
- Include a clear description of what changed and why
- Ensure all CI checks pass
- Add tests for any new functionality

## Running Tests

```bash
make test
# Or directly:
PYTHONPATH=. pytest tests/ -v
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
Run `make lint` to check and auto-fix style issues.
