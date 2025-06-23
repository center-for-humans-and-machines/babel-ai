from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Type, override

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# LLM Interface #
class Provider(Enum):
    """Provider for LLM."""

    OPENAI = "openai"
    OLLAMA = "ollama"

    def get_model_enum(self) -> Type[Enum]:
        """Get the corresponding model enum for this provider."""
        match self:
            case Provider.OPENAI:
                pass
            case Provider.OLLAMA:
                pass
            case _:
                raise ValueError(f"Invalid provider: {self}")

    def get_request_function(self) -> Callable:
        """Get the corresponding request function for this provider."""
        match self:
            case Provider.OPENAI:
                pass
            case Provider.OLLAMA:
                pass
            case _:
                raise ValueError(f"Invalid provider: {self}")


def generate_instruct_response(
    msg_tree: List[Dict[str, str]],
    provider,
    model,
    kwargs,
) -> str:
    # check model is fitting instruct
    # check model and provider are compatible
    # get request function
    # generate response
    pass


def generate_base_response(
    prompt: str,
    provider,
    model,
    kwargs,
) -> str:
    # check model is fitting base
    # check model and provider are compatible
    # get request function
    # generate response
    pass


# Agent #
class AgentConfig(BaseModel):
    """Configuration for an agent."""

    provider: str
    model: str
    temperature: float
    max_tokens: int
    frequency_penalty: float
    presence_penalty: float
    top_p: float


class Agent:
    """An agent that is used in the experiment."""

    def __init__(
        self,
        agent_config: AgentConfig,
    ):
        """Initialize the Agent."""
        self.agent_config = agent_config
        self.provider = Provider(agent_config.provider)
        self.model = self.provider.get_model_enum()(agent_config.model)
        self.model_type = "base" if self.model.is_base_model() else "instruct"
        self.context_size = self.model.get_context_size()

        self.last_msg_tree: List[Dict[str, str]] = []

    def add_roles(self, messages: List[str]) -> List[Dict[str, str]]:
        """Flip the roles of the messages."""
        pass

    def get_msg_tree(
        self,
        messages: List[str],
    ) -> List[Dict[str, str]]:
        """Get the msg tree."""
        # create msg tree
        # cut to context size
        # return msg tree
        pass

    def get_prompt(
        self,
        messages: List[str],
    ) -> str:
        # join messages
        # cut to context size
        # return prompt
        pass

    def generate_response(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Generate a response from the agent."""
        if self.model.is_instruct_model():
            msg_tree = self.get_msg_tree(messages)
            self.last_msg_tree = msg_tree
            response = generate_instruct_response(msg_tree)
        elif self.model.is_base_model():
            prompt = self.get_prompt(messages)
            response = generate_base_response(prompt)
        else:
            raise ValueError(f"Invalid model: {self.model}")

        return response

    def get_model_info(self) -> Dict[str, Any]:
        """Get the model info."""
        info = self.provider.get_model_info(self.model)
        info["model_type"] = self.model_type
        info["context_size"] = self.context_size
        return info


# Analysis #


class AnalysisResult(BaseModel):
    """Analysis results for LLM outputs."""

    word_count: int  # Total number of words in the text
    unique_word_count: int  # Number of unique words
    coherence_score: float  # Ratio of unique words to total words
    lexical_similarity: float  # Jaccard similarity
    # between current and previous text
    semantic_similarity: float  # Cosine similarity between sentence embeddings
    token_perplexity: float  # Perplexity of the text


class Metric(BaseModel):
    """A single metric entry from a drift experiment."""

    iteration: int
    timestamp: datetime
    model_type: Literal["base", "instruct", "initial"]
    agent_config: AgentConfig
    agent_msg_tree: List[Dict[str, str]]
    msg: str
    analysis: AnalysisResult


class AnalyzerConfig(BaseModel):
    """Configuration for the analyzer."""

    analyze_window: int


class Analyzer:
    """Analyzes stats and similarity in LLM outputs."""

    # Semantic similarity model
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    semantic_model_name = semantic_model._get_name()

    # Token model and tokenizer
    token_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    token_model_name = token_model._get_name()

    token_model.eval()

    def __init__(self, analyze_window):
        """Initialize the Analyzer."""
        self.analyze_window = analyze_window

    def analyze(self, texts: List[str]) -> str:
        """Analyze the text."""
        # run all kind of sub methods to get the analysis result
        # word_stats
        results = {}
        results.update(self.analyze_word_stats(texts[-1]))
        results.update(self.analyze_token_perplexity(texts[-1]))

        previous_texts = texts[-self.analyze_window :]
        if previous_texts:
            results.update(self.analyze_lexical_similarity(previous_texts))
            results.update(self.analyze_semantic_similarity(previous_texts))

        return AnalysisResult(**results)


# Prompt Fetcher #


class AnalyzerType(Enum):
    """Analyzer for LLM outputs."""

    SIMILARITY = "similarity"


class Fetcher(Enum):
    """Fetcher for prompts."""

    SHAREGPT = "sharegpt"
    RANDOM = "random"
    INFINITE_CONVERSATION = "infinite_conversation"
    HUMAN_HUMAN = "human_human"


class FetcherConfig(BaseModel):
    """Configuration for the fetcher."""

    data_path: str
    min_messages: int
    max_messages: int


class PromptFetcher:
    """Fetches prompts from various sources."""

    def __init__(self):
        """Initialize the PromptFetcher."""
        pass

    def get_prompt(self, kwargs) -> List[Dict[str, str]]:
        """Get a random prompt history."""
        return [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
        ]

    def create_fetcher(
        self, fetcher_type: Fetcher, **kwargs
    ) -> "PromptFetcher":
        """Create a fetcher from a fetcher type."""
        match fetcher_type:
            case Fetcher.SHAREGPT:
                return SpecificPromptFetcher(**kwargs)
            case Fetcher.RANDOM:
                return SpecificPromptFetcher(**kwargs)
            case Fetcher.INFINITE_CONVERSATION:
                return SpecificPromptFetcher(**kwargs)
            case Fetcher.HUMAN_HUMAN:
                return SpecificPromptFetcher(**kwargs)


class SpecificPromptFetcher(PromptFetcher):
    # Implement for
    # ShareGPT dataset
    # Random prompt
    # Infinite Conversation dataset
    # Human-human dataset

    @override
    def get_prompt(self, kwargs) -> List[Dict[str, str]]:
        """Get a prompt history through a specific method."""
        # run some method to get the message tree from the specific method
        return [
            {"role": "user", "content": "Tell me about specific methods."},
            {
                "role": "assistant",
                "content": "A specific method is a method "
                "that serves a single, well-defined purpose.",
            },
        ]


# Experiment #


class ExperimentConfig(BaseModel):
    """Configuration for the experiment."""

    fetcher: str
    fetcher_config: Dict[str, Any]
    analyzer: str
    analyzer_config: AnalyzerConfig
    agent_configs: List[AgentConfig]
    agent_selection_method: str
    max_iterations: int
    max_total_characters: int


class Experiment:
    """Runs an experiment."""

    def __init__(
        self,
        config: ExperimentConfig,
    ):
        """Initialize the Experiment."""
        self.config = config
        self.prompt_fetcher = PromptFetcher(config.fetcher)
        self.analyzer = Analyzer(config.analyzer_config)
        self.agents = [
            Agent(agent_config) for agent_config in config.agent_configs
        ]

        match config.agent_selection_method:
            case 1:
                self.agent_selection_method_1 = self._get_next_agent
            case 2:
                self.agent_selection_method_2 = self._get_next_agent

        self.starting_msg_tree = self.prompt_fetcher.get_prompt()
        self.current_msgs = list(
            map(lambda x: x["content"], self.starting_msg_tree)
        )

        self.results: List[Metric] = []

    def _get_next_agent(self) -> Agent:
        """Get the next agent."""
        # Use agent selection method to get the next agent
        raise NotImplementedError("Agent selection method not implemented")

    def analyze_messages(
        self,
        messages: List[str],
        model_type: Literal["base", "instruct", "initial"],
        agent_config: AgentConfig,
        agent_msg_tree: List[Dict[str, str]],
    ) -> Metric:
        """Analyze the messages."""
        analysis = self.analyzer.analyze(messages)
        return Metric(
            iteration=len(self.results),
            timestamp=datetime.now(),
            model_type=model_type,
            agent_config=agent_config,
            agent_msg_tree=agent_msg_tree,
            msg=messages[-1],
            analysis=analysis,
        )

    def analyze_initial_messages(self) -> Metric:
        """Analyze the initial messages."""
        return self.analyze_messages(
            messages=self.starting_msg_tree, model_type="initial"
        )

    def run(self):
        """Run the experiment."""
        self.analyze_initial_messages()

        for i in range(self.config.max_iterations):
            self.current_agent = self.agent_selection_method_1()
            response = self.current_agent.generate_response(self.current_msgs)
            metric = self.analyze_messages(
                messages=self.current_msgs,
                model_type=self.current_agent.model_type,
                agent_config=self.current_agent.agent_config,
                agent_msg_tree=self.current_agent.last_msg_tree,
            )
            self.results.append(metric)

            self.current_msgs.append(response)

        return self.results

    def save_results(
        self,
        results: List[Metric],
        output_dir: str,
    ) -> None:
        """Save the results to a CSV file."""
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Run the experiment."""
        results = self.run()
        self.save_results(results, self.config.output_dir)
        return results
