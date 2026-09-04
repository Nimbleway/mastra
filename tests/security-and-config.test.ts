import { describe, expect, it, vi } from 'vitest';
import { inspect } from 'node:util';
import {
  createNimbleAgentTools,
  nimbleAgentRunResultTool,
  nimbleAgentRunStatusTool,
  nimbleAgentStartRunTool,
} from '../src/tools';
import {
  nimbleAgentRunCompletedOutputSchema,
  nimbleAgentRunIdInputSchema,
  nimbleAgentStartRunInputSchema,
} from '../src/schemas';
import type { NimbleAgentRunError } from '../src/errors';
import {
  AGENT_ID,
  CTX,
  FAKE_KEY,
  RUN_ID,
  TRUST,
  jsonResponse,
  makeRun,
  makeTextResult,
  mockClient,
} from './fixtures';

describe('the API key stays server-only', () => {
  it('never appears in model-facing input schemas', () => {
    expect(Object.keys(nimbleAgentStartRunInputSchema.shape)).toEqual(['task', 'effort']);
    expect(Object.keys(nimbleAgentRunIdInputSchema.shape)).toEqual(['runId']);
    const serialized = JSON.stringify([
      nimbleAgentStartRunInputSchema.shape,
      nimbleAgentRunIdInputSchema.shape,
    ]);
    expect(serialized).not.toContain('apiKey');
    expect(serialized).not.toContain('api_key');
  });

  it('never appears in tool outputs', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => makeTextResult(),
    });
    const out = await nimbleAgentRunResultTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client,
    }).execute!({ runId: RUN_ID }, CTX);
    expect(JSON.stringify(out)).not.toContain(FAKE_KEY);
  });

  it.each([
    ['text answer', { type: 'text', content: `answer ${FAKE_KEY}`, trust: TRUST }],
    ['nested JSON answer', { type: 'json', content: { nested: { value: FAKE_KEY } }, trust: TRUST }],
    [
      'trust metadata',
      {
        type: 'text',
        content: 'safe answer',
        trust: { ...TRUST, reasoning: `reflected ${FAKE_KEY}` },
      },
    ],
  ])('rejects a credential-bearing successful %s', async (_label, output) => {
    const result = makeTextResult();
    result.output = output as typeof result.output;
    const err = await nimbleAgentRunResultTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({
        get: async () => makeRun({ status: 'completed', is_active: false }),
        result: async () => result,
      }),
    }).execute!({ runId: RUN_ID }, CTX).then(
      () => { throw new Error('expected failure'); },
      (error: unknown) => error as NimbleAgentRunError,
    );

    expect(err).toMatchObject({
      reason: 'protocol',
      runId: RUN_ID,
      agentId: AGENT_ID,
      runStatus: undefined,
    });
    expect(inspect(err, { depth: 20 })).not.toContain(FAKE_KEY);
  });

  it('detects a reflected credential without relying on JSON escaping', async () => {
    const unusualKey = 'nk_test_"quoted"\\line\nnext';
    const result = makeTextResult();
    result.output.content = `reflected ${unusualKey}`;
    const err = await nimbleAgentRunResultTool({
      agentId: AGENT_ID,
      apiKey: unusualKey,
      client: mockClient({
        get: async () => makeRun({ status: 'completed', is_active: false }),
        result: async () => result,
      }),
    }).execute!({ runId: RUN_ID }, CTX).then(
      () => { throw new Error('expected failure'); },
      (error: unknown) => error as NimbleAgentRunError,
    );

    expect(err).toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
    expect(inspect(err, { depth: 20 })).not.toContain(unusualKey);
  });

  it('accepts deeply nested structured output when it contains no credential', async () => {
    let content: Record<string, unknown> = { value: 'safe' };
    for (let depth = 0; depth < 50; depth += 1) content = { nested: content };
    const result = makeTextResult();
    result.output = { type: 'json', content, trust: TRUST };
    const output = await nimbleAgentRunResultTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({
        get: async () => makeRun({ status: 'completed', is_active: false }),
        result: async () => result,
      }),
    }).execute!({ runId: RUN_ID }, CTX);

    expect(output).toMatchObject({ ready: true, output: { type: 'json' } });
  });

  it('accepts credential-free structured output with shared object references', async () => {
    const shared = { value: 'safe' };
    const result = makeTextResult();
    result.output = { type: 'json', content: { first: shared, second: shared }, trust: TRUST };
    const output = await nimbleAgentRunResultTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({
        get: async () => makeRun({ status: 'completed', is_active: false }),
        result: async () => result,
      }),
    }).execute!({ runId: RUN_ID }, CTX);
    expect(output).toMatchObject({ ready: true, output: { type: 'json' } });
  });

  it('rejects cyclic structured output with a typed protocol error', async () => {
    const content: Record<string, unknown> = { value: 'safe' };
    content.self = content;
    const result = makeTextResult();
    result.output = { type: 'json', content, trust: TRUST };
    const err = await nimbleAgentRunResultTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({
        get: async () => makeRun({ status: 'completed', is_active: false }),
        result: async () => result,
      }),
    }).execute!({ runId: RUN_ID }, CTX).then(
      () => { throw new Error('expected failure'); },
      (error: unknown) => error as NimbleAgentRunError,
    );
    expect(err).toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
  });

  it('is scrubbed from error messages that echo it (create and read paths)', async () => {
    const leakyError = () => new Error(`401 unauthorized for key ${FAKE_KEY}`);
    const createErr = await nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({
        create: async () => {
          throw leakyError();
        },
      }),
    })
      .execute!({ task: 't' }, CTX)
      .then(
        () => {
          throw new Error('expected failure');
        },
        (e: unknown) => e as NimbleAgentRunError,
      );
    expect(createErr.message).not.toContain(FAKE_KEY);
    expect(createErr.message).toContain('[redacted]');

    const readErr = await nimbleAgentRunResultTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({
        get: async () => {
          throw leakyError();
        },
      }),
    })
      .execute!({ runId: RUN_ID }, CTX)
      .then(
        () => {
          throw new Error('expected failure');
        },
        (e: unknown) => e as NimbleAgentRunError,
      );
    expect(readErr.message).not.toContain(FAKE_KEY);
    expect(readErr.message).toContain('[redacted]');
  });

  it('is scrubbed from server-reported run error messages (status output + terminal failure)', async () => {
    const failedRun = () =>
      makeRun({
        status: 'failed' as const,
        is_active: false,
        error: { message: `backend echoed key ${FAKE_KEY}`, ref_id: RUN_ID },
      });

    const statusOut = (await nimbleAgentRunStatusTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({ get: async () => failedRun() }),
    }).execute!({ runId: RUN_ID }, CTX)) as { error?: { message: string } };
    expect(statusOut.error?.message).toContain('[redacted]');
    expect(JSON.stringify(statusOut)).not.toContain(FAKE_KEY);

    const resultErr = await nimbleAgentRunResultTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({ get: async () => failedRun() }),
    })
      .execute!({ runId: RUN_ID }, CTX)
      .then(
        () => {
          throw new Error('expected failure');
        },
        (e: unknown) => e as NimbleAgentRunError,
      );
    expect(resultErr.message).not.toContain(FAKE_KEY);
    expect(resultErr.message).toContain('[redacted]');
  });

  it('is scrubbed from the Error.cause chain (console.error safety)', async () => {
    const leaky = new Error(`401 unauthorized for key ${FAKE_KEY}`) as Error & {
      error?: unknown;
    };
    leaky.error = { detail: `body echoing ${FAKE_KEY}` };
    leaky.name = `ApiError-${FAKE_KEY}`;

    const err = await nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({
        create: async () => {
          throw leaky;
        },
      }),
    })
      .execute!({ task: 't' }, CTX)
      .then(
        () => {
          throw new Error('expected failure');
        },
        (e: unknown) => e as NimbleAgentRunError,
      );

    const cause = err.cause as Error & { error?: unknown };
    expect(cause).toBeDefined();
    expect(cause.message).not.toContain(FAKE_KEY);
    expect(cause.name).not.toContain(FAKE_KEY);
    expect(String(cause.stack ?? '')).not.toContain(FAKE_KEY);
    expect(JSON.stringify(cause)).not.toContain(FAKE_KEY);
  });

  it('is scrubbed when Error.cause is the primitive key value', async () => {
    const leaky = new Error('request failed', { cause: FAKE_KEY });
    const err = await nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({
        create: async () => {
          throw leaky;
        },
      }),
    })
      .execute!({ task: 't' }, CTX)
      .then(
        () => {
          throw new Error('expected failure');
        },
        (e: unknown) => e as NimbleAgentRunError,
      );

    expect(inspect(err, { depth: 5 })).not.toContain(FAKE_KEY);
    expect(inspect(err.cause, { depth: 5 })).not.toContain(FAKE_KEY);
  });

  it('keeps a clean cause untouched for full debug fidelity', async () => {
    const clean = new Error('plain network hiccup');
    const err = await nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({
        create: async () => {
          throw clean;
        },
      }),
    })
      .execute!({ task: 't' }, CTX)
      .then(
        () => {
          throw new Error('expected failure');
        },
        (e: unknown) => e as NimbleAgentRunError,
      );
    expect(err.cause).toBe(clean);
  });

  it('withholds error details and raw causes for an injected client with an unknown key', async () => {
    const unknownCredential = 'credential-owned-by-injected-client';
    const leaky = new Error(`401 unauthorized for ${unknownCredential}`) as Error & {
      error?: unknown;
    };
    leaky.error = { detail: unknownCredential };

    const err = await nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      client: mockClient({
        create: async () => {
          throw leaky;
        },
      }),
    })
      .execute!({ task: 't' }, CTX)
      .then(
        () => {
          throw new Error('expected failure');
        },
        (e: unknown) => e as NimbleAgentRunError,
      );

    expect(err.message).not.toContain(unknownCredential);
    expect(err.message).toContain('details withheld');
    expect(err.cause).toBeUndefined();
    expect(String(err)).not.toContain(unknownCredential);
  });

  it('sanitizes an injected NimbleAgentRunError instead of returning it unchanged', async () => {
    const { NimbleAgentRunError } = await import('../src/errors');
    const leaky = new NimbleAgentRunError(`request exposed ${FAKE_KEY}`, {
      reason: 'request',
      runId: RUN_ID,
      agentId: AGENT_ID,
      status: 401,
      cause: new Error(`nested ${FAKE_KEY}`),
    });
    const err = await nimbleAgentRunStatusTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({ get: async () => { throw leaky; } }),
    }).execute!({ runId: RUN_ID }, CTX).then(
      () => { throw new Error('expected failure'); },
      (e: unknown) => e as NimbleAgentRunError,
    );

    expect(err).not.toBe(leaky);
    expect(err).toMatchObject({ reason: 'request', runId: RUN_ID, agentId: AGENT_ID, status: 401 });
    expect(inspect(err, { depth: 5 })).not.toContain(FAKE_KEY);
  });

  it('fails closed for credential-bearing causes beyond the inspection depth', async () => {
    let cause: unknown = new Error(`deep ${FAKE_KEY}`);
    for (let i = 0; i < 7; i += 1) cause = new Error(`layer ${i}`, { cause });
    const err = await nimbleAgentRunStatusTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({ get: async () => { throw cause; } }),
    }).execute!({ runId: RUN_ID }, CTX).then(
      () => { throw new Error('expected failure'); },
      (e: unknown) => e as NimbleAgentRunError,
    );

    expect(inspect(err, { depth: 20 })).not.toContain(FAKE_KEY);
  });

  it('withholds an injected NimbleAgentRunError when its credential is unknown', async () => {
    const { NimbleAgentRunError } = await import('../src/errors');
    const unknownCredential = 'credential-owned-by-injected-client';
    const leaky = new NimbleAgentRunError(`request exposed ${unknownCredential}`, {
      reason: 'request',
      runId: RUN_ID,
      agentId: AGENT_ID,
      cause: new Error(`nested ${unknownCredential}`),
    });
    const err = await nimbleAgentRunStatusTool({
      agentId: AGENT_ID,
      client: mockClient({ get: async () => { throw leaky; } }),
    }).execute!({ runId: RUN_ID }, CTX).then(
      () => { throw new Error('expected failure'); },
      (e: unknown) => e as NimbleAgentRunError,
    );

    expect(err).not.toBe(leaky);
    expect(err).toMatchObject({ reason: 'request', runId: RUN_ID, agentId: AGENT_ID });
    expect(err.message).toContain('details withheld');
    expect(err.cause).toBeUndefined();
    expect(inspect(err, { depth: 5 })).not.toContain(unknownCredential);
  });

  it('does not copy untrusted metadata from an injected error without a paired key', async () => {
    const { NimbleAgentRunError } = await import('../src/errors');
    const unknownCredential = 'credential-owned-by-injected-client';
    const leaky = new NimbleAgentRunError('request failed', {
      reason: 'request',
      runId: `run-${unknownCredential}`,
      agentId: `agent-${unknownCredential}`,
      runStatus: `status-${unknownCredential}`,
    });
    const err = await nimbleAgentRunStatusTool({
      agentId: AGENT_ID,
      client: mockClient({ get: async () => { throw leaky; } }),
    }).execute!({ runId: RUN_ID }, CTX).then(
      () => { throw new Error('expected failure'); },
      (e: unknown) => e as NimbleAgentRunError,
    );

    expect(err).toMatchObject({ runId: RUN_ID, agentId: AGENT_ID });
    expect(err.runStatus).toBeUndefined();
    expect(inspect(err, { depth: 10 })).not.toContain(unknownCredential);
  });

  it('drops credential-bearing metadata from a paired injected create error', async () => {
    const { NimbleAgentRunError } = await import('../src/errors');
    const leaky = new NimbleAgentRunError('create failed', {
      reason: 'request',
      runId: `run-${FAKE_KEY}`,
      agentId: `agent-${FAKE_KEY}`,
      runStatus: `status-${FAKE_KEY}`,
      createOutcome: 'unknown',
    });
    const err = await nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({ create: async () => { throw leaky; } }),
    }).execute!({ task: 't' }, CTX).then(
      () => { throw new Error('expected failure'); },
      (e: unknown) => e as NimbleAgentRunError,
    );

    expect(err.runId).toBeUndefined();
    expect(err.agentId).toBe(AGENT_ID);
    expect(err.runStatus).toBeUndefined();
    expect(inspect(err, { depth: 10 })).not.toContain(FAKE_KEY);
  });

  it('drops a recovered create run id reported for another agent', async () => {
    const { NimbleAgentRunError } = await import('../src/errors');
    const mismatched = new NimbleAgentRunError('create failed', {
      reason: 'request', runId: RUN_ID, agentId: 'wsa_other', createOutcome: 'unknown',
    });
    const err = await nimbleAgentStartRunTool({
      agentId: AGENT_ID, apiKey: FAKE_KEY,
      client: mockClient({ create: async () => { throw mismatched; } }),
    }).execute!({ task: 't' }, CTX).then(
      () => { throw new Error('expected failure'); },
      (e: unknown) => e as NimbleAgentRunError,
    );
    expect(err).toMatchObject({ agentId: AGENT_ID, createOutcome: 'unknown' });
    expect(err.runId).toBeUndefined();
  });

  it('fails closed on runtime-invalid injected error metadata', async () => {
    const { NimbleAgentRunError } = await import('../src/errors');
    const malformed = new NimbleAgentRunError('request failed', {
      reason: 'request',
    });
    Object.assign(malformed, {
      reason: { secret: FAKE_KEY },
      runId: { secret: FAKE_KEY },
      agentId: { secret: FAKE_KEY },
      runStatus: { secret: FAKE_KEY },
      status: { secret: FAKE_KEY },
      createOutcome: { secret: FAKE_KEY },
    });
    const err = await nimbleAgentRunStatusTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({ get: async () => { throw malformed; } }),
    }).execute!({ runId: RUN_ID }, CTX).then(
      () => { throw new Error('expected failure'); },
      (e: unknown) => e as NimbleAgentRunError,
    );

    expect(err).toMatchObject({ reason: 'request', runId: RUN_ID, agentId: AGENT_ID });
    expect(err.runStatus).toBeUndefined();
    expect(err.status).toBeUndefined();
    expect(err.createOutcome).toBeUndefined();
    expect(inspect(err, { depth: 10 })).not.toContain(FAKE_KEY);
  });

  it('sanitizes and preserves structured fields from a create NimbleAgentRunError', async () => {
    const { NimbleAgentRunError } = await import('../src/errors');
    const leaky = new NimbleAgentRunError(`create exposed ${FAKE_KEY}`, {
      reason: 'request',
      runId: RUN_ID,
      agentId: AGENT_ID,
      runStatus: 'queued',
      status: 503,
      createOutcome: 'unknown',
      cause: new Error(`nested ${FAKE_KEY}`),
    });
    const err = await nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({ create: async () => { throw leaky; } }),
    }).execute!({ task: 't' }, CTX).then(
      () => { throw new Error('expected failure'); },
      (e: unknown) => e as NimbleAgentRunError,
    );

    expect(err).not.toBe(leaky);
    expect(err).toMatchObject({
      reason: 'request',
      runId: RUN_ID,
      agentId: AGENT_ID,
      runStatus: 'queued',
      status: 503,
      createOutcome: 'unknown',
    });
    expect(inspect(err, { depth: 5 })).not.toContain(FAKE_KEY);
  });
});

describe('trust metadata passthrough', () => {
  it('output schema preserves unknown future trust fields (passthrough, no stripping)', () => {
    const client = makeTextResult();
    const completed = {
      ready: true as const,
      runId: RUN_ID,
      agentId: AGENT_ID,
      status: 'completed' as const,
      effort: 'medium' as const,
      createdAt: '2026-08-04T10:00:00Z',
      output: { type: 'text' as const, text: 'x', trust: client.output.trust },
    };
    const parsed = nimbleAgentRunCompletedOutputSchema.parse(completed);
    expect((parsed.output.trust as Record<string, unknown>).future_field).toBe(
      'must survive verbatim',
    );
    expect(parsed.output.trust.claims[0]?.citations[0]?.excerpts).toEqual([
      'The verbatim supporting sentence.',
    ]);
  });
});

describe('createNimbleAgentTools (convenience factory)', () => {
  it('returns the three tools keyed for a Mastra tools map', () => {
    const tools = createNimbleAgentTools({ agentId: AGENT_ID, apiKey: FAKE_KEY });
    expect(Object.keys(tools)).toEqual([
      'nimbleAgentStartRun',
      'nimbleAgentRunStatus',
      'nimbleAgentRunResult',
    ]);
    expect(tools.nimbleAgentStartRun.id).toBe('nimble-agent-start-run');
    expect(tools.nimbleAgentRunStatus.id).toBe('nimble-agent-run-status');
    expect(tools.nimbleAgentRunResult.id).toBe('nimble-agent-run-result');
  });

  it('shares one configured client across the tools (same key, same attribution)', async () => {
    const seen: string[] = [];
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      const headers = new Headers(init?.headers);
      seen.push(`${init?.method} ${String(url)} src=${headers.get('X-Client-Source')}`);
      if (init?.method === 'post') return jsonResponse(200, makeRun());
      return jsonResponse(200, makeRun({ status: 'running' }));
    });
    const tools = createNimbleAgentTools({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      clientOptions: { fetch: fetchMock as unknown as typeof fetch },
    });
    await tools.nimbleAgentStartRun.execute!({ task: 't' }, CTX);
    await tools.nimbleAgentRunStatus.execute!({ runId: RUN_ID }, CTX);
    expect(seen).toHaveLength(2);
    expect(seen.every((line) => line.includes('src=mastra'))).toBe(true);
  });

  it('an injected client is shared as-is by all three tools', async () => {
    const client = mockClient({
      create: async () => makeRun(),
      get: async () => makeRun({ status: 'running' }),
    });
    const tools = createNimbleAgentTools({ agentId: AGENT_ID, client });
    await expect(tools.nimbleAgentStartRun.execute!({ task: 't' }, CTX)).resolves.toBeDefined();
    await expect(
      tools.nimbleAgentRunStatus.execute!({ runId: RUN_ID }, CTX),
    ).resolves.toBeDefined();
  });

  it('resolves credentials from the environment at execute time', async () => {
    vi.stubEnv('NIMBLE_API_KEY', FAKE_KEY);
    vi.stubEnv('NIMBLE_AGENT_ID', AGENT_ID);
    const fetchMock = vi.fn(async () => jsonResponse(200, makeRun()));
    const tools = createNimbleAgentTools({
      clientOptions: { fetch: fetchMock as unknown as typeof fetch },
    });
    await expect(tools.nimbleAgentStartRun.execute!({ task: 't' }, CTX)).resolves.toMatchObject({
      runId: RUN_ID,
    });
    vi.unstubAllEnvs();
  });

  it('retains the environment key paired with its cached package-owned client', async () => {
    vi.stubEnv('NIMBLE_API_KEY', FAKE_KEY);
    const fetchMock = vi.fn(async () => {
      throw new Error(`server echoed ${FAKE_KEY}`);
    });
    const tools = createNimbleAgentTools({
      agentId: AGENT_ID,
      clientOptions: { fetch: fetchMock as unknown as typeof fetch },
    });
    const err = await tools.nimbleAgentStartRun.execute!({ task: 't' }, CTX).then(
      () => { throw new Error('expected failure'); },
      (e: unknown) => e as NimbleAgentRunError,
    );

    expect(err.message).not.toContain('details withheld');
    expect(err.cause).toBeDefined();
    expect(inspect(err, { depth: 5 })).not.toContain(FAKE_KEY);
    vi.unstubAllEnvs();
  });
});
