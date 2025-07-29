# Experiments and Notebooks Guide

## Running Experiments

### Basic Experiment Execution

#### 1. Command Line Interface
```bash
# Basic experiment with default settings
poetry run python src/main.py configs/test_config.yaml

# With debug logging
poetry run python src/main.py configs/test_config.yaml --debug

# Sequential execution (easier debugging)
poetry run python src/main.py configs/test_config.yaml --sequential

# Multiple configurations
poetry run python src/main.py config1.yaml config2.yaml config3.yaml
```

#### 2. Programmatic Usage
```python
import asyncio
from babel_ai.experiment import Experiment, ExperimentConfig
from utils import load_yaml_config

# Load configuration
config = load_yaml_config(ExperimentConfig, "configs/test_config.yaml")

# Run single experiment
experiment = Experiment(config)
results = experiment.run()

# Run multiple experiments
configs = [load_yaml_config(ExperimentConfig, path)
          for path in ["config1.yaml", "config2.yaml"]]

async def run_batch():
    from main import run_experiment_batch
    await run_experiment_batch(configs, parallel=True)

asyncio.run(run_batch())
```

### Configuration Details

#### Basic Configuration Structure
```yaml
# configs/example_config.yaml

# Data source configuration
fetcher_config:
  fetcher: "sharegpt"
  data_path: "data/sharegpt_sample.json"
  min_messages: 2
  max_messages: 10

# Analysis configuration
analyzer_config:
  analyzer: "similarity"
  analyze_window: 5

# Agent configurations (can be multiple)
agent_configs:
  - provider: "openai"
    model: "gpt-4-1106-preview"
    system_prompt: "You are a helpful assistant participating in a conversation."
    temperature: 0.7
    max_tokens: 150
    frequency_penalty: 0.0
    presence_penalty: 0.0
    top_p: 1.0

# Experiment parameters
agent_selection_method: "round_robin"
max_iterations: 50
max_total_characters: 100000
output_dir: "results"
```

#### Available Fetchers

**ShareGPT Fetcher:**
```yaml
fetcher_config:
  fetcher: "sharegpt"
  data_path: "data/sharegpt_conversations.json"
  min_messages: 2
  max_messages: 15
```

**Topical Chat Fetcher:**
```yaml
fetcher_config:
  fetcher: "topical_chat"
  data_path: "data/train.json"
  second_data_path: "data/valid.json"  # Optional validation set
  min_messages: 3
  max_messages: 20
```

**Infinite Conversation Fetcher:**
```yaml
fetcher_config:
  fetcher: "infinite_conversation"
  data_path: "data/infinite_conversations.json"
  min_messages: 5
  max_messages: 25
```

**Random Prompt Fetcher:**
```yaml
fetcher_config:
  fetcher: "random"
  category: "creative"  # Options: creative, analytical, conversational
```

#### Available Analyzers

**Similarity Analyzer (Primary):**
```yaml
analyzer_config:
  analyzer: "similarity"
  analyze_window: 5  # Rolling window size for similarity comparisons
```

This analyzer calculates:
- Semantic similarity using sentence transformers
- Lexical similarity via token overlap
- Token perplexity using GPT-2
- Text coherence scores
- Word count and uniqueness metrics

#### Agent Configuration Options

**OpenAI Provider:**
```yaml
agent_configs:
  - provider: "openai"
    model: "gpt-4-1106-preview"  # or "gpt-3.5-turbo", "gpt-4-turbo"
    system_prompt: "Your system prompt here"
    temperature: 0.7
    max_tokens: 150
    frequency_penalty: 0.0
    presence_penalty: 0.0
    top_p: 1.0
```

**Anthropic Provider:**
```yaml
agent_configs:
  - provider: "anthropic"
    model: "claude-3-haiku-20240307"
    system_prompt: "Your system prompt here"
    temperature: 0.7
    max_tokens: 150
```

**Azure OpenAI Provider:**
```yaml
agent_configs:
  - provider: "azure_openai"
    model: "gpt-35-turbo"  # Your deployment name
    system_prompt: "Your system prompt here"
    temperature: 0.7
    max_tokens: 150
```

**Ollama Provider (Local):**
```yaml
agent_configs:
  - provider: "ollama"
    model: "llama2"  # or "mistral", "codellama", etc.
    system_prompt: "Your system prompt here"
    temperature: 0.7
    max_tokens: 150
```

#### Multi-Agent Configurations

**Multiple Agents with Different Providers:**
```yaml
agent_configs:
  - provider: "openai"
    model: "gpt-4-1106-preview"
    system_prompt: "You are an optimistic conversationalist."
    temperature: 0.8

  - provider: "anthropic"
    model: "claude-3-haiku-20240307"
    system_prompt: "You are a analytical conversationalist."
    temperature: 0.3

agent_selection_method: "round_robin"  # or "random"
```

### Environment Setup

#### API Keys Configuration
Create `.env` file in project root:
```bash
# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Optional: Weights & Biases (for tracking)
WANDB_API_KEY=your_wandb_key_here
```

### Understanding Results

#### Output Structure
```
results/
├── experiment_20240115_143022_<uuid>.csv  # Detailed metrics
└── experiment_20240115_143022_<uuid>.json # Metadata
```

#### CSV Metrics Columns
```
timestamp,iteration,agent_id,provider,model,message_content,
word_count,unique_word_count,coherence_score,token_perplexity,
lexical_similarity,semantic_similarity,lexical_similarity_window,
semantic_similarity_window,total_characters,response_time_ms
```

#### JSON Metadata
```json
{
  "timestamp": "2024-01-15T14:30:22",
  "config": { ... },  // Full experiment configuration
  "num_fetcher_messages": 3,
  "total_iterations": 50,
  "total_characters": 8542,
  "agents": [ ... ],  // Agent configurations
  "experiment_id": "uuid-here"
}
```
