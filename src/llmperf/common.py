from typing import List
from llmperf.ray_clients.litellm_client import LiteLLMClient
from llmperf.ray_clients.lookahead_client import LookaheadClient
from llmperf.ray_clients.nvcf_client import NVCFChatCompletionsClient
from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.ray_clients.sagemaker_client import SageMakerClient
from llmperf.ray_clients.vertexai_client import VertexAIClient
from llmperf.ray_llm_client import LLMClient


SUPPORTED_APIS = ["openai", "anthropic", "litellm", "lookahead", "nvcf"]


def construct_clients(llm_api: str, num_clients: int) -> List[LLMClient]:
    """Construct LLMClients that will be used to make requests to the LLM API.

    Args:
        llm_api: The name of the LLM API to use.
        num_clients: The number of concurrent requests to make.

    Returns:
        The constructed LLMCLients

    """
    if llm_api == "openai":
        clients = [OpenAIChatCompletionsClient.remote() for _ in range(num_clients)]
    elif llm_api == "sagemaker":
        clients = [SageMakerClient.remote() for _ in range(num_clients)]
    elif llm_api == "vertexai":
        clients = [VertexAIClient.remote() for _ in range(num_clients)]
    elif llm_api == "lookahead":
        clients = [LookaheadClient.remote() for _ in range(num_clients)]
    elif llm_api == "nvcf":
        clients = [NVCFChatCompletionsClient.remote() for _ in range(num_clients)]
    elif llm_api in SUPPORTED_APIS:
        clients = [LiteLLMClient.remote() for _ in range(num_clients)]
    else:
        raise ValueError(
            f"llm_api must be one of the supported LLM APIs: {SUPPORTED_APIS}"
        )

    return clients
