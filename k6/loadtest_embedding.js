import http from 'k6/http';
import { check, sleep } from 'k6';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';
import exec from 'k6/execution';

export let options = {
    stages: [
        { duration: '10s', target: 1 },
        { duration: '1m', target: 1 },
        { duration: '10s', target: 2 },
        { duration: '1m', target: 2 },
        { duration: '10s', target: 4 },
        { duration: '1m', target: 4 },
        { duration: '10s', target: 8 },
        { duration: '1m', target: 8 },
        { duration: '10s', target: 16 },
        { duration: '1m', target: 16 },
        { duration: '10s', target: 32 },
        { duration: '1m', target: 32 },
        { duration: '10s', target: 64 },
        { duration: '1m', target: 64 },
        { duration: '10s', target: 128 },
        { duration: '1m', target: 128 },
        { duration: '10s', target: 0 },
    ],
    summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(95)', 'count'],
};

// Define prompts
const prompts = [
    "If Partners Asia has been doing aid differently for over 20 years, but took a 3-year break at some point, and the article was published in 2023, what is the latest year they could have started their operations including the break period?",
    "How would you apply the principles of using 'a' and 'the' in contract language to draft a clause that specifies the delivery of goods in a sales contract, provided that the goods must first pass a pre-delivery inspection to confirm adherence to the agreed-upon quality standards?",
    "How many states expanded their Medicaid eligibility systems to include all Medicaid groups and at least one non-health program by 2016, and how does this compare to the number of states with such expansions before the implementation of the Affordable Care Act?",
    'Using syntax analysis, write a regular expression for identifying a valid e-mail address in Python.',
    "Could you search for 'fitness routines' with a timeout of 3000ms and also check the status of job ID 'VJ345678'?",
    'Explain the core concept of correlation analysis and its limitations in understanding relationships between variables in a dataset.',
    'In a Media portal system, how would you implement a feature that notifies users with a yellow caution triangle icon 30 days before their account is scheduled to be deactivated due to inactivity of 180 days? Describe the steps involved in the implementation process.',
    'You are an expert in the field of text classifier. You need to classify all texts below, each into one of the given classes. \n\nExplain your reasoning step by step.\n\nTexts:\n\n[\'In response to the recent influenza outbreak, our healthcare facility is reviewing the shift patterns of our nursing staff to ensure continuous patient care.\', "In the wake of a traumatic brain injury, the individual\'s altered neuromuscular function demands a synergistic therapeutic strategy, encompassing both neuroplasticity-promoting exercises and the utilization of adaptive equipment to augment ambulatory stability and mitigate the risk of secondary complications.", \'Given the recent patent expiration of a popular antihypertensive medication, the surge in demand for the generic version has outpaced production capabilities, compelling healthcare providers to develop a protocol for equitable distribution among patients with varying degrees of cardiovascular risk, while also considering the ethical implications of such decisions.\', \'The victim, exhibiting signs of a cerebrovascular accident with left-sided paralysis and aphasia, urgently needs neuroimaging and potential thrombolytic therapy to address the acute ischemic stroke and prevent irreversible neurological damage.\', \'An individual with severe mental health challenges, who is frequently in and out of psychiatric care, needs a structured support system that bridges the gap between clinical treatment and community living. This involves coordinating mental health services, peer support groups, housing assistance, and employment opportunities to foster a stable and supportive environment for recovery.\', \'Upon admission, the individual was experiencing profound dissociative episodes, coupled with intense paranoia, which precipitated the need for rapid tranquilization and the implementation of a high-dependency monitoring protocol to circumvent potential self-injurious behavior.\']\n\nClasses:\n\nPsychiatric Crisis Intervention - Immediate Stabilization;Chronic Psychiatric Condition Management;Emergency Medical Response - Critical Intervention;Chronic Disease Management - Long-term Stabilization;Advanced Life Support - Critical Care Interventions;Postoperative Critical Care Monitoring and Management;Infectious Risk Mitigation - Prophylactic Strategies;Infectious Disease Management - Active Therapeutic Intervention;Equipment Utilization - Resource Optimization;Staffing Optimization - Personnel Deployment;Pharmacological Resource Prioritization;Functional Rehabilitation - Mobility and Ambulation Support;Nutritional Management - Specialized Feeding Support;Social Support Coordination - Community Care Integration;Multifaceted Presentation - Indeterminate Primary Need;Data Deficiency - Inadequate Information for Classification\n\nThe output format must be:\n\n{"results": [{"text": text, "class": class, "reasoning": reasoning}, {"text": text, "class": class, "reasoning": reasoning}, {"text": text, "class": class, "reasoning": reasoning}, {"text": text, "class": class, "reasoning": reasoning}]}',
    'You are tasked with writing a Python function `at_pose` that checks whether robots are "close enough" to their desired poses. This is an important function in robotics for determining if a robot has reached its target position and orientation. The function will utilize several numpy library functions to compute position and rotation errors and decide if the criteria for being "close enough" are met.\n\nThe function signature is as follows:\n```python\ndef at_pose(states, poses, position_error=0.05, rotation_error=0.2):\n    """\n    Checks whether robots are "close enough" to poses\n\n    Parameters\n    ----------\n    states : numpy.ndarray\n        3xN array of unicycle states [x, y, theta]\n    poses : numpy.ndarray\n        3xN array of desired states [x, y, theta]\n    position_error : float, optional\n        Allowable position error (default is 0.05)\n    rotation_error : float, optional\n        Allowable angular error (default is 0.2)\n\n    Returns\n    -------\n    done : numpy.ndarray\n        1xN index array of agents that are close enough\n    """\n```\n\nYour task involves:\n- Validating the input parameters.\n- Calculating position and rotation errors between the current states and the desired poses.\n- Determining which robots are within the allowable position and rotation errors.\n\nLibraries to use:\n- `numpy.abs`\n- `numpy.arctan2`\n- `numpy.sin`\n- `numpy.nonzero`\n\n**Constraints and Assumptions:**\n1. The input arrays `states` and `poses` are numpy arrays of shape (3, N), where N is the number of robots.\n2. Position errors are calculated using Euclidean distance.\n3. Rotation errors are calculated considering the shortest angular distance.\n\n**Example:**\n```python\nimport numpy as np\n\nstates = np.array([[1, 2], [1, 2], [0.1, 0.5]])\nposes = np.array([[1, 2], [1.01, 2.1], [0.15, 0.55]])\nat_pose(states, poses, position_error=0.1, rotation_error=0.1)\n```\nExpected Output:\n```\n(array([0, 1]),)\n```\n\n###',
    'In the context of computational biology, how can the X-DEE algorithm be utilized to analyze the ensemble of protonation states in bacteriorhodopsin, and what insights might this provide into the functional mechanism of this protein?'
]

const API_URL = 'http://10.67.27.26:8000/v1/embeddings';
const MODEL = 'nvidia/llama-3.2-nv-embedqa-1b-v2';

export default function () {
    // Add stage logging
    if (exec.scenario.stage) {
        console.log(`Current stage: ${exec.scenario.stage.name}, Target VUs: ${exec.scenario.stage.target}`);
    }

    // Select a random prompt
    let selectedPrompt = randomItem(prompts);

    let payload = JSON.stringify({
        model: MODEL,
        input: selectedPrompt,
        input_type: 'query',
        encoding_format: 'float',
        dimensions: 384,
        truncate: 'NONE'
    });

    let params = {
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        tags: {
            stage: `${__VU} VUs`
        }
    };

    let response = http.post(API_URL, payload, params);

    // Validate response
    check(response, {
        'is status 200': (r) => r.status === 200,
        'response contains usage': (r) => JSON.parse(r.body).usage !== undefined,
    });

    // Add response time logging
    console.log(`Response time: ${response.timings.duration}ms`);

    // Log usage (optional)
    // let usage = JSON.parse(response.body).usage;
    // console.log(`Usage: ${JSON.stringify(usage)}`);
}