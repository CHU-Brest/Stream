# Contributing to Stream

Thank you for your interest in contributing to Stream! This guide will help you get started with development, testing, and submitting changes.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Poetry (recommended) or pip
- Git

### Development Environment Setup

#### Option 1: Using Poetry (Recommended)

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/CHU-Brest/Stream.git
cd Stream

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/CHU-Brest/Stream.git
cd Stream

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
# Generate synthetic medical reports using the Brest pipeline
python cli.py brest --n-sejours 100

# Generate using the AP-HP pipeline
python cli.py aphp --n-sejours 100
```

## 📝 Coding Guidelines

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style.
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all public functions and classes.
- Use type hints for all function signatures and variables where applicable.

### Linting and Formatting

We use the following tools to enforce code quality:

- **`black`**: Code formatter
- **`flake8`**: Linter
- **`mypy`**: Static type checker

To run these tools:

```bash
# Format code with black
black .

# Lint code with flake8
flake8 .

# Check types with mypy
mypy .
```

### Commit Messages

- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.
- Examples:
  - `feat: add new pipeline for XYZ data source`
  - `fix: handle empty data in get_fictive`
  - `docs: update README with installation instructions`
  - `refactor: decompose pipeline into smaller modules`

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_pipelines.py -v

# Run tests with coverage
pytest --cov=pipelines tests/ -v
```

### Writing Tests

- Follow the [pytest](https://docs.pytest.org/) framework conventions.
- Test both happy paths and edge cases.
- Use mocks for external dependencies (e.g., LLM APIs, file I/O).

### Test Coverage

Aim for **>= 80% test coverage** for all new code. Use `pytest-cov` to measure coverage:

```bash
pytest --cov=pipelines --cov-report=html tests/
```

## 📦 Pull Request Process

### Before Submitting a PR

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Ensure your code passes all checks**:
   ```bash
   black .
   flake8 .
   mypy .
   pytest tests/ -v
   ```

3. **Update documentation** if your changes affect the API or usage.

4. **Rebase your branch** onto the latest `main` branch:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

### Submitting a PR

1. Push your branch to your fork:
   ```bash
   git push origin feat/my-feature
   ```

2. Open a **Pull Request** against the `main` branch of the upstream repository.

3. Fill in the PR template with:
   - A clear description of the changes
   - Related issues (if any)
   - Screenshots (if UI changes)

4. Wait for review and address any feedback.

### Review Process

- All PRs require **at least one approval** from a maintainer.
- PRs will be merged using **squash merge** to keep history clean.
- Once approved, a maintainer will merge your PR.

## 📚 Documentation

### Updating Documentation

- Update `README.md` for user-facing changes.
- Update `TODOS.md` for planned features/improvements.
- Update `CONTRIBUTING.md` if contribution guidelines change.

### Generating API Docs

We use [Sphinx](https://www.sphinx-doc.org/) to generate API documentation:

```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Generate docs
cd docs
make html
```

## 🤝 Code of Conduct

By participating in this project, you agree to abide by the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## 📫 Contact

For questions or discussions, please open an [issue](https://github.com/CHU-Brest/Stream/issues) or join our [Discussions](https://github.com/CHU-Brest/Stream/discussions).

---

*Last updated: 2024-04-06*
