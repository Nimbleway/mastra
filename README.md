# @nimble-way/mastra

Nimble deep-research **Agent runs** ([Agent API V2](https://docs.nimbleway.com)) as
ready-made, typed tools for [Mastra](https://mastra.ai) agents.

A Nimble Web Search Agent run researches the live web for minutes and returns an
answer (prose or structured JSON) with **trust metadata**: sources, per-claim
citations with verbatim excerpts, and confidence. This package exposes that
lifecycle to a Mastra agent as three non-blocking tools:

| Tool id | What it does |
|---|---|
| `nimble-agent-start-run` | Starts a run on your configured agent; returns the `task_run_…` ID immediately |
| `nimble-agent-run-status` | Instant status snapshot: `queued` / `running` / `completed` / `failed` / `cancelled` |
| `nimble-agent-run-result` | Fetches the finished answer, or `{ ready: false }` while the run is still working |

Runs are **resumable**: any process configured with the same agent can check or
collect a run it did not start — the `runId` is the only handle needed. A
multi-minute research run is never hidden inside a blocking tool call.

## Install

Requires Node.js 22.13 or later and Mastra 1.56 or later within the supported
1.x release line.

```bash
npm install @nimble-way/mastra @mastra/core zod
```

## Quickstart

```ts
import { Agent } from '@mastra/core/agent';
import { createNimbleAgentTools } from '@nimble-way/mastra';

const researcher = new Agent({
  id: 'researcher',
  name: 'Researcher',
  instructions:
    'For deep research questions, start a Nimble agent run, tell the user the ' +
    'runId, and fetch the result when they ask for it later.',
  model: 'anthropic/claude-sonnet-4.6',
  // One shared API key + agent configuration for all three tools:
  tools: createNimbleAgentTools({
    agentId: process.env.NIMBLE_AGENT_ID, // wsa_… instance, created once in the Nimble console
  }),
});
```

Configuration is resolved server-side at execute time: `NIMBLE_API_KEY` and
`NIMBLE_AGENT_ID` from the environment, or `apiKey` / `agentId` passed to the
factory. **The API key never appears in tool schemas, tool outputs, or model
input** — the model only ever chooses the research task, an optional effort
tier, and the runId to look up.

Individual factories are also exported when you want per-tool configuration:

```ts
import {
  nimbleAgentStartRunTool,
  nimbleAgentRunStatusTool,
  nimbleAgentRunResultTool,
} from '@nimble-way/mastra';

const tools = {
  startResearch: nimbleAgentStartRunTool({ effortCap: 'medium' }),
  researchStatus: nimbleAgentRunStatusTool(),
  researchResult: nimbleAgentRunResultTool(),
};
```

## The resumable lifecycle

```ts
// Turn 1 — the agent starts a run and answers immediately:
//   start → { runId: 'task_run_…', status: 'queued', … }

// Turn 2 (seconds or hours later, same or different process):
//   status → { status: 'running', isActive: true, … }

// Turn 3:
//   result → { ready: false, status: 'running', … }   // still working — not an error
//   result → { ready: true, status: 'completed', output: { type: 'text', text, trust } }
```

If you poll outside the agent loop, a **10-second interval** is the recommended
default — runs take minutes; polling faster only spends requests:

```ts
const result = nimbleAgentRunResultTool({
  wait: { pollIntervalMs: 10_000, timeoutMs: 120_000 },
});
```

`wait` is opt-in and bounded: when the timeout elapses the tool returns
`{ ready: false }` and the run keeps going server-side — check again later.
Without `wait` (the default) the result tool never blocks.

## Answers and trust

A completed run returns either prose or structured data, plus verbatim trust
metadata:

```ts
{
  ready: true,
  status: 'completed',
  runId: 'task_run_…',
  output: {
    type: 'text',                    // or 'json' with `json` instead of `text`
    text: 'The answer … [1]',
    trust: {
      confidence: 'high',
      reasoning: '…',
      sources: [{ url, type: 'primary', source_category: 'official', … }],
      claims: [{ callout: 1, confidence, reasoning, citations: [{ url, excerpts, … }] }],
    },
  },
}
```

`trust` is passed through verbatim (snake_case preserved, unknown future fields
kept) so citation markers stay aligned with the answer text and nothing is
silently dropped.

## Error semantics

- **Run creation is one-shot.** The POST is sent with `maxRetries: 0` — a run
  is billed and not idempotent, so the SDK's transient-retry policy (408/409/
  429/5xx) is disabled for creation only. Status and result reads keep normal
  retries.
- When creation fails, the thrown `NimbleAgentRunError` carries
  `createOutcome`:
  - `'not-created'` (definite 4xx such as 429): no run exists; retry only after
    the underlying condition clears.
  - `'unknown'` (timeout, 408, 409, 5xx, connection drop): the run **may** have
    been created server-side. Do not automatically start another run — list
    recent runs for the agent (or check the Nimble console) and reconcile first.
- A run that terminally `failed` / `cancelled` throws `NimbleAgentRunError`
  with `reason`, the server's message, and the `runId` preserved so the run can
  still be inspected or reported.
- A still-active run is **never** an error: the result tool returns
  `{ ready: false }` (including when the result endpoint answers HTTP 409
  while the status read already said `completed`).
- Out-of-contract payloads surface as `reason: 'protocol'` errors, not
  TypeErrors.

## Effort control

The model may request an effort tier (`low` → `max`), clamped to the
developer-configured `effortCap` (default `high`), so a model cannot
unilaterally trigger the most expensive tiers. The developer `effort` default
is not clamped; leaving both unset uses the agent instance's own configured
default.

## Security

- `NIMBLE_API_KEY` is read server-side at execute time and sent only as a
  request header. It is scrubbed from error messages and never present in tool
  schemas, outputs, fixtures, or logs.
- Every request this package makes carries `X-Client-Source: mastra`.

## License

Apache-2.0
