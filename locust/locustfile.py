import random
from locust import HttpUser, task, between

PROMPT = [
    "What is machine learning?",
    "Explain the process of natural language processing.",
    "How do neural networks work?",
    "What are embedding vectors used for?",
    "Describe the transformer architecture.",
    "What is the difference between supervised and unsupervised learning?",
    "How does BERT generate embeddings?",
    "What is transfer learning in NLP?",
    "Explain the concept of attention in neural networks.",
    "How are embeddings evaluated for quality?"
]

class EmbeddingLoadTest(HttpUser):
    wait_time = between(1, 2)

    @task
    def embed_task(self):
        prompt_idx = random.choice(PROMPT)
        self.client.post(
            "/v1/embeddings",
            json={
                "model": "BAAI/bge-m3",
                "input": prompt_idx,
            },
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
        )
