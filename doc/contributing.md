# Contributing Guide

## Development Setup

### Prerequisites
- Python 3.13+
- Poetry (dependency management)
- Git

### Initial Setup
1. **Fork and clone**:
   ```bash
   git clone <your-fork-url>
   cd babel_ai
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   poetry shell
   ```

3. **Install pre-commit hooks**:
   ```bash
   poetry run pre-commit install
   ```

4. **Environment setup**:
   ```bash
   # Create .env file with API keys (for integration tests)
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Code Standards

### Formatting
- **Line length**: 79 characters maximum
- **Formatter**: Black with project configuration
- **Import sorting**: isort with Black profile
- **Type hints**: Required for all public functions

### Code Style
- Follow functional programming patterns where beneficial
- Use list comprehensions and pattern matching when appropriate
- Minimize side effects in utility functions
- Prefer composition over inheritance

### Example:
```python
def analyze_similarity(
    texts: List[str], window_size: int = 5
) -> Optional[float]:
    """Analyze semantic similarity between texts.

    Args:
        texts: List of text strings to analyze
        window_size: Rolling window size for comparison

    Returns:
        Similarity score or None if insufficient data
    """
    if len(texts) < window_size:
        return None

    return sum(
        calculate_similarity(texts[i], texts[i + 1])
        for i in range(len(texts) - 1)
    ) / (len(texts) - 1)
```

## Testing Requirements

### Test Structure
```
tests/
├── unit_tests/
│   ├── api_tests/         # API provider tests
│   ├── prompt_fetchers/   # Data fetcher tests
│   └── test_*.py          # Core module tests
└── integration/           # End-to-end tests
```

### Testing Guidelines
1. **Coverage**: Maintain >85% test coverage
2. **Test types**:
   - Unit tests for individual functions/classes
   - Integration tests for full workflows
   - Mocked API calls (no real API usage in tests)

3. **Naming**: Use descriptive test names
   ```python
   def test_similarity_analyzer_with_insufficient_data():
   def test_experiment_saves_results_to_correct_directory():
   ```

### Running Tests
```bash
# All tests
poetry run pytest

# With coverage
poetry run coverage run -m pytest
poetry run coverage report
poetry run coverage html  # Generate HTML report

# Specific test types
poetry run pytest tests/unit_tests/
poetry run pytest tests/integration/
poetry run pytest -k "similarity"  # Run tests matching pattern
```

### Writing Tests
- **Concise**: Cover essential functionality only
- **Fast**: Unit tests should run quickly
- **Isolated**: No dependencies between tests
- **Deterministic**: Same input = same output

Example test:
```python
def test_agent_generates_response():
    """Test that agent generates valid response."""
    config = AgentConfig(
        provider=Provider.OPENAI,
        model=OpenAIModels.GPT_4,
        system_prompt="Test prompt"
    )

    with patch('api.openai.OpenAIAPI.generate_response') as mock:
        mock.return_value = "Test response"
        agent = Agent(config)

        response = agent.generate_response([
            {"role": "user", "content": "Hello"}
        ])

        assert response == "Test response"
        mock.assert_called_once()
```

## Development Workflow

### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# Write tests
# Run tests and formatting

# Commit with descriptive message
git commit -m "feat: add semantic similarity analyzer

- Implement cosine similarity calculation
- Add rolling window analysis
- Include comprehensive tests"
```

### 2. Before Committing
```bash
# Format code
poetry pre-commit run --all-files
```

### 3. Pull Request Process
1. Ensure all tests pass
2. Update documentation if needed
3. Add test coverage for new features
4. Use descriptive PR title and description
5. Link to relevant issues
