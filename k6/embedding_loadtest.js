import http from 'k6/http';
import { check, sleep } from 'k6';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';
import exec from 'k6/execution';

// Configuration
const API_URL = 'http://10.67.27.1:8000/v1/embeddings';
const MODEL = 'BAAI/bge-m3';

// Sample prompts - replace with your actual prompts
const prompts = [
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
];

// K6 test configuration
export const options = {
    scenarios: {
        embedding_load_test: {
            executor: 'ramping-vus',
            startVUs: 1,
            stages: [
                { duration: '30s', target: 1 },
                { duration: '30s', target: 4 },
                { duration: '1m', target: 16 },
                { duration: '1m', target: 32 },
                { duration: '1m', target: 64 },
                { duration: '1m', target: 4 },
                { duration: '30s', target: 0 },
            ],
            gracefulRampDown: '30s',
        },
    },
    thresholds: {
        http_req_duration: ['p(95)<500'], // 95% of requests should complete within 500ms
        http_req_failed: ['rate<0.01'],   // Less than 1% of requests should fail
    },
};

export default function () {
    // Add stage and concurrency logging
    const currentVUs = exec.vu.idInInstance;
    const currentStage = exec.scenario.iterationInTest > 0 ?
        exec.scenario.name + ' - ' + currentVUs + ' VUs' :
        'Starting test';

    console.log(`Current concurrency level: ${currentVUs} VUs`);

    // Select a random prompt
    let selectedPrompt = randomItem(prompts);

    // Prepare request payload
    let payload = JSON.stringify({
        model: MODEL,
        input: selectedPrompt,
        encoding_format: 'float',
        truncate_prompt_tokens: 1
    });

    // Set request parameters
    let params = {
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        tags: {
            concurrency: `${currentVUs} VUs`,
            stage: currentStage
        }
    };

    // Send request and measure
    const startTime = new Date().getTime();
    let response = http.post(API_URL, payload, params);
    const endTime = new Date().getTime();
    const duration = endTime - startTime;

    // Check response
    check(response, {
        'status is 200': (r) => r.status === 200,
        'has embedding data': (r) => r.json().hasOwnProperty('data'),
    });

    // Log response time with concurrency information
    console.log(`Request completed in ${duration}ms at concurrency level: ${currentVUs} VUs`);

    // Add a small sleep to prevent overwhelming the system
    sleep(0.5);
}

// Custom metrics - grouped by concurrency level
export function handleSummary(data) {
    // Group metrics by concurrency level
    const concurrencyMetrics = {};

    if (data.metrics && data.metrics.http_req_duration) {
        Object.keys(data.metrics.http_req_duration.values).forEach(key => {
            if (key.startsWith('concurrency:')) {
                const concurrencyLevel = key.split(':')[1];
                concurrencyMetrics[concurrencyLevel] = {
                    avg: data.metrics.http_req_duration.values[key].avg,
                    min: data.metrics.http_req_duration.values[key].min,
                    med: data.metrics.http_req_duration.values[key].med,
                    max: data.metrics.http_req_duration.values[key].max,
                    p90: data.metrics.http_req_duration.values[key].p(90),
                    p95: data.metrics.http_req_duration.values[key].p(95),
                    p99: data.metrics.http_req_duration.values[key].p(99),
                };
            }
        });
    }

    return {
        'summary.json': JSON.stringify({
            metrics: {
                http_req_duration: data.metrics.http_req_duration.values,
            },
            concurrencyMetrics,
        }, null, 2),
        'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    };
}