import http from 'k6/http';
import { check, sleep } from 'k6';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

export let options = {
    stages: [
        { duration: '10s', target: 10 },
        { duration: '1m', target: 16 },
        { duration: '10s', target: 0 },
    ],
};

// Define prompts
const prompts = [
    `Large pre-trained language models have been shown to store factual knowledge... Provide appropriate title of the above content.`,
    `Write a few paragraphs for the technical writeup around Nvidia NVAIE s/w stack offering in 1000-1500 words.`,
    `Large pre-trained language models have been shown to store factual knowledge... Provide summary of the above content in 200 words.`
];

const API_URL = 'https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/f06a2c59-3d8d-4bfb-bd3f-44f3235b0541';
const API_KEY = '';

export default function () {
    // Select a random prompt
    let selectedPrompt = randomItem(prompts);

    let payload = JSON.stringify({
        model: '/workspace/model_onprem/',
        messages: [
            { role: 'user', content: 'Hello! How are you?' },
            { role: 'assistant', content: 'Hi! I am quite well, how can I help you today?' },
            { role: 'user', content: selectedPrompt }
        ],
        temperature: 0.8,
        max_tokens: 2048,
        // stream: false
    });

    let params = {
        headers: {
            'Authorization': `Bearer ${API_KEY}`,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
    };

    let response = http.post(API_URL, payload, params);

    // Validate response
    check(response, {
        'is status 200': (r) => r.status === 200,
        'response contains usage': (r) => JSON.parse(r.body).usage !== undefined,
    });

    // Log usage (optional)
    let usage = JSON.parse(response.body).usage;
    console.log(`Usage: ${JSON.stringify(usage)}`);

    // sleep(1);  // Wait between requests
}