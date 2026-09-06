import type {
  NimbleAgentRawFailedResult,
  NimbleAgentRawResult,
  NimbleAgentRawRun,
  NimbleAgentRunsClient,
  NimbleAgentTrust,
} from '../src/schemas';

export const AGENT_ID = 'wsa_11111111-2222-3333-4444-555555555555';
export const RUN_ID = 'task_run_aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';
export const FAKE_KEY = 'nk_test_fixture_key_never_real_1234567890';

export function makeRun(overrides: Partial<NimbleAgentRawRun> = {}): NimbleAgentRawRun {
  return {
    id: RUN_ID,
    interaction_id: 'interaction_0001',
    status: 'queued',
    is_active: true,
    effort: 'medium',
    created_at: '2026-08-04T10:00:00Z',
    web_search_agent_id: AGENT_ID,
    ...overrides,
  };
}

/** Trust fixture incl. an unknown future field to prove verbatim passthrough. */
export const TRUST: NimbleAgentTrust = {
  confidence: 'high',
  reasoning: 'Multiple primary sources agree.',
  sources: [
    {
      url: 'https://example.gov/report',
      type: 'primary',
      title: 'Official report',
      source_category: 'official',
    },
  ],
  claims: [
    {
      confidence: 'high',
      reasoning: 'Directly stated by the primary source.',
      callout: 1,
      citations: [
        {
          url: 'https://example.gov/report',
          title: 'Official report',
          excerpts: ['The verbatim supporting sentence.'],
          source_type: 'primary',
          source_category: 'official',
        },
      ],
    },
  ],
  // Unknown-future-field probe (allowed by passthrough):
  future_field: 'must survive verbatim',
} as NimbleAgentTrust;

export function makeTextResult(overrides: Partial<NimbleAgentRawRun> = {}): NimbleAgentRawResult {
  return {
    run: makeRun({ status: 'completed', is_active: false, completed_at: '2026-08-04T10:05:00Z', ...overrides }),
    output: { type: 'text', content: 'The answer [1].', trust: TRUST },
  };
}

export function makeJsonResult(): NimbleAgentRawResult {
  return {
    run: makeRun({ status: 'completed', is_active: false }),
    output: { type: 'json', content: { rows: [{ name: 'x' }] }, trust: TRUST },
  };
}

export function makeFailedResult(message = 'Run failed upstream'): NimbleAgentRawFailedResult {
  return {
    run: makeRun({ status: 'failed', is_active: false, error: { message, ref_id: RUN_ID } }),
    error: { message, ref_id: RUN_ID },
  };
}

type RunsSlice = NimbleAgentRunsClient['agents']['runs'];

/** Build an injected mock client from partial method implementations. */
export function mockClient(impl: Partial<RunsSlice>): NimbleAgentRunsClient {
  const unexpected = (name: string) => async () => {
    throw new Error(`unexpected call to runs.${name}`);
  };
  return {
    agents: {
      runs: {
        create: impl.create ?? (unexpected('create') as RunsSlice['create']),
        get: impl.get ?? (unexpected('get') as RunsSlice['get']),
        result: impl.result ?? (unexpected('result') as RunsSlice['result']),
      },
    },
  };
}

/** An HTTP-shaped SDK error (matches Stainless APIError's `status`/`error`). */
export function httpError(status: number, body?: unknown): Error & { status: number; error?: unknown } {
  const err = new Error(`${status} status error`) as Error & { status: number; error?: unknown };
  err.status = status;
  if (body !== undefined) err.error = body;
  return err;
}

/** Minimal Response factory for fetch-level (real SDK client) tests. */
export function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

/** Minimal execute context for direct tool.execute calls in tests. */
export const CTX = {} as never;
