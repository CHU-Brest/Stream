# TODOS — Stream Project Roadmap

This document outlines the planned improvements and features for the Stream project.
Each section corresponds to a specific area of improvement, with detailed tasks and estimated timelines.

## 📋 Current Priorities

### 1. **Modularity & Separation of Concerns** *(Week 1-2)*
- [ ] **Decompose pipelines into smaller, specialized modules**
  - [ ] Create `fictive.py` for fictitious stay generation
  - [ ] Create `scenario.py` for scenario transformation
  - [ ] Create `report.py` for CRH generation
  - [ ] Define clear interfaces between modules
  - [ ] Document module responsibilities in docstrings

- [ ] **Refactor AP-HP pipeline to use internal modules** *(follow-up to current refactoring)*
  - [ ] Integrate `loader`, `scenario`, `managment`, `prompt`, `sampler` directly into `pipeline.py`
  - [ ] Ensure backward compatibility with existing code
  - [ ] Update imports and references

### 2. **Testing Improvements** *(Week 2-3)*
- [ ] **Add integration tests**
  - [ ] Test Brest + AP-HP pipeline interactions
  - [ ] Test pipeline + LLM client interactions (mocked)
  - [ ] Test data loading and validation workflows

- [ ] **Add performance tests**
  - [ ] Benchmark pipeline execution with varying data volumes
  - [ ] Measure memory usage and optimize data structures
  - [ ] Add CI/CD performance regression checks

- [ ] **Enhance unit tests**
  - [ ] Increase coverage for edge cases (empty data, invalid configs)
  - [ ] Add property-based testing (e.g., Hypothesis) for data generators
  - [ ] Mock external dependencies (LLM APIs, file I/O) for faster tests

### 3. **Documentation** *(Week 3-4)*
- [ ] **Complete docstrings for all public functions/classes**
  - [ ] Follow [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
  - [ ] Add type hints to all functions
  - [ ] Document exceptions raised

- [ ] **Generate API documentation with Sphinx**
  - [ ] Set up Sphinx with autodoc
  - [ ] Add narrative documentation (tutorials, examples)
  - [ ] Host documentation on GitHub Pages/ReadTheDocs

- [ ] **Expand README.md**
  - [ ] Add architecture diagrams (Mermaid.js)
  - [ ] Add "Getting Started" guide with concrete examples
  - [ ] Document common pitfalls and solutions

### 4. **Performance Optimization** *(Week 4-5)*
- [ ] **Optimize data operations**
  - [ ] Profile hotspots with `cProfile`/`py-spy`
  - [ ] Replace inefficient Pandas operations with Polars
  - [ ] Use lazy evaluation where possible

- [ ] **Parallelize operations**
  - [ ] Parallelize `get_fictive` across GHM groups
  - [ ] Use `multiprocessing` or `ray` for CPU-bound tasks
  - [ ] Batch LLM calls where possible

- [ ] **Optimize I/O**
  - [ ] Use Parquet for intermediate data (instead of CSV)
  - [ ] Compress large data files
  - [ ] Cache loaded referentials in memory

### 5. **Code Quality & Maintainability** *(Ongoing)*
- [ ] **Enforce code style**
  - [ ] Configure `black` (formatting)
  - [ ] Configure `flake8` (linting)
  - [ ] Configure `mypy` (type checking)
  - [ ] Add pre-commit hooks

- [ ] **Add CI/CD checks**
  - [ ] Run linters in GitHub Actions
  - [ ] Enforce test coverage thresholds
  - [ ] Add automated documentation builds

- [ ] **Improve error handling**
  - [ ] Define custom exception hierarchy
  - [ ] Add context to error messages (e.g., "Failed to load {file}")
  - [ ] Log errors with structured logging (JSON)

### 6. **Configuration & Validation** *(Week 5-6)*
- [ ] **Enhance configuration validation**
  - [ ] Use `pydantic` for config validation
  - [ ] Add schema validation for YAML files
  - [ ] Provide clear error messages for invalid configs

- [ ] **Support multiple config formats**
  - [ ] Add support for JSON/TOML configs
  - [ ] Allow environment variable overrides
  - [ ] Add config examples for different use cases

### 7. **Developer Experience** *(Week 6-7)*
- [ ] **Add CONTRIBUTING.md** *(This file!)*
  - [x] Development environment setup
  - [x] Coding guidelines
  - [x] Testing guidelines
  - [x] Pull request process

- [ ] **Improve logging**
  - [ ] Add structured logging (JSON)
  - [ ] Log key metrics (e.g., stays generated, time taken)
  - [ ] Add log levels (DEBUG/INFO/WARNING/ERROR)

- [ ] **Add GitHub templates**
  - [ ] Pull request template
  - [ ] Issue templates (bug report, feature request)
  - [ ] Add labels for issue triage

### 8. **Advanced Features** *(Future)*
- [ ] **Add plugin system** for custom pipelines
- [ ] **Support streaming generation** for large datasets
- [ ] **Add monitoring/dashboards** (Prometheus/Grafana)
- [ ] **Add CLI autocompletion** (argcomplete)

## 📅 Timeline

| Phase | Duration | Focus Area |
|-------|----------|------------|
| 1 | 2 weeks | Modularity, Separation of Concerns |
| 2 | 2 weeks | Testing (Integration, Performance) |
| 3 | 2 weeks | Documentation (Docstrings, Sphinx) |
| 4 | 2 weeks | Performance Optimization |
| 5 | Ongoing | Code Quality (Linting, CI/CD) |
| 6 | 2 weeks | Configuration & Validation |
| 7 | 2 weeks | Developer Experience |

## 🎯 Goals

- **Modularity**: Make it easy to add new pipelines or modify existing ones
- **Reliability**: Ensure pipelines handle edge cases gracefully
- **Performance**: Optimize for large-scale data generation
- **Maintainability**: Keep code clean, documented, and easy to understand
- **Accessibility**: Lower the barrier for new contributors

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

---

*Last updated: 2024-04-06*
*Status: Active*
