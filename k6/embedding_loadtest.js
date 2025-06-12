import http from 'k6/http';
import { check, sleep } from 'k6';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Configuration
const API_URL = 'http://10.67.27.1:8800/v1/embeddings';
const API_KEY = '';
const MODEL = 'BAAI/bge-multilingual-gemma2';

const prompts = [
    `You're a policy advisor tasked with developing a comprehensive urban development strategy for rapidly urbanizing Tier-2 cities in India. Your focus areas include sustainable infrastructure, affordable housing, multimodal public transportation, walkability, green open spaces, water conservation, and waste management. Incorporate provisions for zoning, mixed-use development, and urban resilience in the face of climate change. Reference and align with national policies like AMRUT, PMAY, and the Smart Cities Mission. Consider stakeholder engagement strategies and mechanisms for public-private partnerships. Provide a phased implementation plan, budget considerations, and mechanisms for monitoring outcomes.`,

    `As a systems engineer working on LLM inference at scale, write a detailed technical proposal addressing latency and throughput issues in a multi-GPU setup. Highlight the architectural bottlenecks, such as memory bandwidth and kernel launch overhead. Propose optimization techniques such as quantization-aware inference, operator fusion, weight prefetching, speculative decoding, and CUDA kernel tuning. Consider using frameworks like vLLM, FasterTransformer, and DeepSpeed-Inference. Define performance benchmarks (latency per token, tokens/sec/GPU, memory utilization) and explain how you'd track regressions in production. Discuss tradeoffs of precision (FP16 vs INT8) and batching strategies for dynamic vs static prompts.`,

    `Write a blog-style case study on building a product recommendation engine using unsupervised learning. The client is an e-commerce brand struggling with cold-start problems and poor personalization. Walk through how you used item embeddings via Word2Vec or autoencoders trained on user-item interactions, replacing collaborative filtering methods. Explain your preprocessing pipeline, how you handled sparse data, and how the model was trained, validated, and deployed. Detail the backend serving stack, latency optimization, and A/B testing results. Share business impact metrics (e.g., CTR uplift, basket size), challenges faced, lessons learned, and post-launch iterations.`,

    `Draft a fictional dialogue between a senior C++ systems engineer and a junior dev discussing how to reduce memory overhead in an LLM inference engine. They should explore placement new, object pools, arena allocators, and how alignment/padding affects memory fragmentation. Include brief code snippets where appropriate. The senior should also explain how to analyze cache misses, avoid false sharing, and use tools like Valgrind, perf, or cachegrind. Keep the tone educational but conversational, simulating a mentoring session inside a performance-obsessed ML systems team.`,

    `You are planning a Transit-Oriented Development (TOD) strategy for the Kerala West Coast Canal corridor, specifically Reach 10. Draft a comprehensive plan covering station-area design, walkability, NMT access, last-mile integration, zoning changes, mixed-use development, and economic uplift potential. Incorporate learnings from TOD implementations in Ahmedabad, Kochi Metro, or Singapore’s MRT. Identify the canal-side constraints, including ecological sensitivity, local livelihoods (fishing, water tourism), and flood risk zones. Propose phased development models, policy reforms needed (e.g., transferable development rights, land value capture), and institutional frameworks for implementation.`,

    `Design a distributed training setup for a 30-billion parameter LLM using PyTorch and DeepSpeed. The document should include GPU/TPU selection (A100 vs H100), network topology (NVLink, InfiniBand), tensor parallelism vs pipeline parallelism, optimizer choice (AdamW vs ZeRO), and training precision (FP16/bfloat16). Cover data pipeline architecture with sharded datasets, tokenization strategy, and checkpointing for fault tolerance. Include estimated training time, compute cost, and cloud vs on-prem tradeoffs. Outline your monitoring strategy using tools like WandB, Prometheus, or NVIDIA Nsight.`,

    `You're the CPO of a startup note-taking app. Write an internal memo announcing a major product pivot: integrating AI summarization, visual mind mapping, and real-time cross-device sync. Justify this move using competitive analysis (Notion, Obsidian, Reflect), user research findings, and product-market fit signals. Explain how AI capabilities (LLM summarization, semantic search, auto-tagging) will enhance user experience. Detail short-term implementation milestones for each team: design (UX flows, prototypes), backend (API changes, DB models), frontend (React refactors), and marketing (launch messaging). Include deadlines and KPIs.`,

    `Write a short story set in 2040 in southern India, where rising sea levels have displaced thousands. A Tamil family is forced to migrate to a “climate-resilient smart city” in Karnataka. Describe their struggle adjusting to new biometric housing policies, AI-based employment matching, and cultural displacement. Use emotional depth to explore themes of loss, resilience, and inequality. Show both the promise and the dehumanizing aspects of a tech-driven society that solved climate infrastructure but not social equity. End with a personal moment between family members that highlights adaptation amid uncertainty.`,

    `Create a technical design doc for implementing a Kubernetes-native CI/CD pipeline in a fintech company. The company wants secure, fast, auditable deployments. Use GitLab CI, Helm, ArgoCD, and Vault. Detail your Helm chart structure, GitOps workflows, RBAC configuration, secrets management, pod security policies, and how rollbacks are handled with ArgoCD. Include how feature branches are promoted, how staging mirrors prod, and how load testing is integrated pre-prod. Discuss compliance logging and integration with an audit dashboard. End with a diagram showing end-to-end flow.`,

    `Compare three open-source inference engines for LLMs: vLLM, TGI (Text Generation Inference), and FasterTransformer. Include setup complexity, support for continuous batching, tokenizer compatibility, GPU memory efficiency, streaming token support, and throughput benchmarks. Provide pros and cons of each for various use cases: chatbot APIs, summarization pipelines, or multi-tenant LLM endpoints. Recommend tools based on requirements like lowest latency, highest throughput, or lowest cost per 1M tokens. Include a summary table comparing key features.`
];


// K6 test configuration
export const options = {
    scenarios: {
        embedding_load_test: {
            executor: 'ramping-vus',
            startVUs: 1,
            stages: [
                { duration: '15s', target: 64 },
                { duration: '2m', target: 64 },
                { duration: '15s', target: 0 },
            ],
            gracefulRampDown: '30s',
        },
    }
};

export default function () {
    let selectedPrompt = randomItem(prompts);

    let payload = JSON.stringify({
        model: MODEL,
        input: selectedPrompt,
        encoding_format: 'float',
        truncate_prompt_tokens: 1
    });

    let params = {
        headers: {
            'Authorization': `Bearer ${API_KEY}`,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    };

    let response = http.post(API_URL, payload, params);

    // Check response
    check(response, {
        'status is 200': (r) => r.status === 200,
        'has embedding data': (r) => {
            try {
                let jsonResponse = JSON.parse(r.body);
                let embedding = jsonResponse.data[0].embedding;
                return embedding !== undefined;
            } catch (e) {
                console.error('Failed to parse response body:', r.body);
                return false;
            }
        },
    });

    sleep(1);
}
