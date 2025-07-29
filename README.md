# babel-ai

A research framework for analyzing long-term behavior and drift patterns in Large Language Models (LLMs) during self-loop conversations without external input.

## Overview

This project investigates whether AIs need external input to create sensible output and explores the importance of human interaction for AI performance. It analyzes how AI interaction can be made more diverse without external input and what productive input patterns look like.

## Quick Start

### Installation

1. **Prerequisites**: Python 3.13+ and Poetry

   **Install Poetry if needed**:
   ```bash
   # Official installer (recommended)
   curl -sSL https://install.python-poetry.org | python3 -

   # Or via pip
   pip install --user poetry

   # Or via Homebrew (macOS)
   brew install poetry
   ```

2. **Clone and install**:
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd babel_ai

   # Install dependencies using Poetry
   poetry install

   # Activate the virtual environment
   poetry shell
   ```

3. **Set up environment** (optional):
   ```bash
   # Create .env file for API keys if using external LLM providers
   cp .env.example .env  # Edit with your API keys
   ```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run coverage run --source=src -m pytest
poetry run coverage report

# Run specific test categories
poetry run pytest tests/unit_tests/     # Unit tests only
poetry run pytest tests/integration/   # Integration tests only
```

### Quick Experiment

1. **Basic experiment**:
   ```bash
   # Run with the provided test configuration
   poetry run python src/main.py configs/test_config.yaml
   ```

2. **Custom experiment**:
   ```bash
   # Create your own config file (see configs/test_config.yaml as template)
   poetry run python src/main.py your_config.yaml --debug
   ```

3. **View results**:
   - Results saved to `results/` directory as CSV and JSON files
   - Use notebooks in `notebooks/` for analysis and visualization

**For detailed experiment configuration, multi-agent setups, and analysis workflows, see [doc/experiments.md](doc/experiments.md).**

## Key Features

- **Multi-Provider Support**: OpenAI, Anthropic, Azure OpenAI, Ollama
- **Drift Analysis**: Semantic similarity, lexical analysis, perplexity metrics
- **Data Sources**: ShareGPT, Topical Chat, Infinite Conversation datasets
- **Configurable Experiments**: YAML-based configuration system
- **Analysis Notebooks**: Jupyter notebooks for result visualization
- **Comprehensive Testing**: Unit and integration test suites

## Project Structure

```
babel_ai/
├── src/
│   ├── babel_ai/          # Core experiment framework
│   ├── api/               # LLM provider interfaces
│   ├── models/            # Data models and configurations
│   └── main.py            # Main experiment runner
├── tests/                 # Test suites
├── notebooks/             # Analysis and visualization
├── configs/               # Experiment configurations
├── doc/                   # Detailed documentation
└── data/                  # Datasets (not included)
```

## Contributing

We welcome contributions! Please see [doc/contributing.md](doc/contributing.md) for detailed guidelines including:
- Code style and formatting (Black, 79-char limit)
- Testing requirements (pytest, coverage)
- Development workflow
- Adding new analyzers, fetchers, or providers

## Documentation

- **[Module Guide](doc/modules.md)**: Detailed explanation of core modules
- **[Experiments Guide](doc/experiments.md)**: How to run experiments and use notebooks
- **[Directory Structure](doc/directory-structure.md)**: Complete project organization
- **[Contributing Guide](doc/contributing.md)**: Development guidelines

## License

[LICENSE](LICENSE)

## Research Context

This project explores fundamental questions about AI behavior:
- Do AIs require external input for coherent output?
- How important is human interaction for AI performance?
- Can AI interaction diversity be improved without external input?
- What input patterns are most productive for AI systems?
