// HTTP floor measurement: hits /ready (no body, no predict) at the same
// ramping load as the real prévia. p99 of this is the network/parse
// overhead budget — subtract from real prévia's p99 to estimate the
// compute share of the latency.
import http from 'k6/http';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

export const options = {
    summaryTrendStats: ['p(50)', 'p(90)', 'p(99)'],
    systemTags: ['status', 'method'],
    scenarios: {
        default: {
            executor: 'ramping-arrival-rate',
            startRate: 1,
            timeUnit: '1s',
            preAllocatedVUs: 100,
            maxVUs: 250,
            gracefulStop: '10s',
            stages: [
                { duration: '120s', target: 900 },
            ],
        },
    },
};

export default function () {
    http.get('http://localhost:9999/ready', { timeout: '2001ms' });
}

export function handleSummary(data) {
    const d = data.metrics.http_req_duration.values;
    const out = {
        p50: `${d['p(50)'].toFixed(2)}ms`,
        p90: `${d['p(90)'].toFixed(2)}ms`,
        p99: `${d['p(99)'].toFixed(2)}ms`,
        count: data.metrics.http_reqs.values.count,
        rate_per_s: data.metrics.http_reqs.values.rate.toFixed(0),
    };
    return {
        stdout: textSummary(data, { indent: '  ', enableColors: false }) + '\n' + JSON.stringify(out, null, 2) + '\n',
    };
}
