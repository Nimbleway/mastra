import { describe, expect, it, vi } from 'vitest';
import { inspect } from 'node:util';
import {
  nimbleAgentRunResultTool,
  nimbleAgentRunStatusTool,
  nimbleAgentStartRunTool,
} from '../src/tools';
import { NimbleAgentRunError, NimbleConfigError } from '../src/errors';
import type { NimbleAgentRunCompletedOutput, NimbleAgentRunCreateBody } from '../src/schemas';
import { nimbleAgentTrustSchema } from '../src/schemas';
import {
  AGENT_ID,
  CTX,
  FAKE_KEY,
  RUN_ID,
  httpError,
  makeFailedResult,
  makeJsonResult,
  makeRun,
  makeTextResult,
  mockClient,
} from './fixtures';

const TEST_API_KEY = 'credential-free-test-key';

describe('start tool', () => {
  it('returns the authoritative server ids', async () => {
    const client = mockClient({ create: async () => makeRun() });
    const out = await nimbleAgentStartRunTool({ agentId: AGENT_ID, client }).execute!(
      { task: 't' },
      CTX,
    );
    expect(out).toEqual({
      runId: RUN_ID,
      agentId: AGENT_ID,
      interactionId: 'interaction_0001',
      status: 'queued',
      effort: 'medium',
      createdAt: '2026-08-04T10:00:00Z',
    });
  });

  it('rejects a create response with a missing authoritative agent id', async () => {
    const client = mockClient({
      create: async () => makeRun({ web_search_agent_id: '' }),
    });
    await expect(
      nimbleAgentStartRunTool({ agentId: AGENT_ID, client }).execute!(
        { task: 't' },
        CTX,
      ),
    ).rejects.toMatchObject({ reason: 'protocol' });
  });

  it('fails closed when create returns a different agent id than requested', async () => {
    const client = mockClient({ create: async () => makeRun() });
    await expect(
      nimbleAgentStartRunTool({ agentId: 'wsa_99999999-8888-7777-6666-555555555555', client }).execute!(
        { task: 't' },
        CTX,
      ),
    ).rejects.toMatchObject({ reason: 'protocol', agentId: 'wsa_99999999-8888-7777-6666-555555555555' });
  });

  it('clamps model-chosen effort to the cap but not developer effort', async () => {
    const bodies: NimbleAgentRunCreateBody[] = [];
    const client = mockClient({
      create: async (_id, body) => {
        bodies.push(body);
        return makeRun();
      },
    });
    // Model asks for max → clamped to default cap (high).
    await nimbleAgentRunToolExec(client, { task: 't', effort: 'max' });
    // Model asks for low → within cap, unchanged.
    await nimbleAgentRunToolExec(client, { task: 't', effort: 'low' });
    expect(bodies.map((b) => b.effort)).toEqual(['high', 'low']);

    // Developer effort is not clamped and applies when the model chose none.
    const devTool = nimbleAgentStartRunTool({ agentId: AGENT_ID, client, effort: 'max' });
    await devTool.execute!({ task: 't' }, CTX);
    expect(bodies[2]?.effort).toBe('max');

    async function nimbleAgentRunToolExec(
      c: typeof client,
      input: { task: string; effort?: 'low' | 'max' },
    ) {
      await nimbleAgentStartRunTool({ agentId: AGENT_ID, client: c }).execute!(input, CTX);
    }
  });

  it('sends developer-owned output schema and non-empty source allowlist', async () => {
    const bodies: NimbleAgentRunCreateBody[] = [];
    const client = mockClient({
      create: async (_id, body) => {
        bodies.push(body);
        return makeRun();
      },
    });
    const outputSchema = {
      type: 'object',
      required: ['facts', 'recommendation'],
      properties: {
        facts: {
          type: 'array',
          description: 'Exactly two facts; validate exact cardinality after retrieval.',
          items: { type: 'object' },
        },
        recommendation: { type: 'string' },
      },
    };
    const sources = {
      allow: [
        { title: 'Mastra repository', domains: ['github.com'], order: 0 },
        { title: 'Mastra documentation', domains: ['mastra.ai'], order: 1 },
      ],
    };

    await nimbleAgentStartRunTool({ agentId: AGENT_ID, client, outputSchema, sources }).execute!(
      { task: 'research Mastra', effort: 'low' },
      CTX,
    );

    expect(bodies).toEqual([
      { input: 'research Mastra', effort: 'low', output_schema: outputSchema, sources },
    ]);
  });

  it('rejects configured empty sources.allow before create', async () => {
    const create = vi.fn(async () => makeRun());
    const client = mockClient({ create });
    const tool = nimbleAgentStartRunTool({ agentId: AGENT_ID, client, sources: { allow: [] } });

    await expect(tool.execute!({ task: 'research Mastra' }, CTX)).rejects.toThrow(
      'sources.allow must contain at least one named source group',
    );
    expect(create).not.toHaveBeenCalled();
  });

  it('rejects a source allow group with no domains before create', async () => {
    const create = vi.fn(async () => makeRun());
    const client = mockClient({ create });
    const tool = nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      client,
      sources: { allow: [{ title: 'Official docs', domains: [] }] },
    });

    await expect(tool.execute!({ task: 'research Mastra' }, CTX)).rejects.toThrow(
      'source group with non-empty domains',
    );
    expect(create).not.toHaveBeenCalled();
  });

  it('rejects an unknown lifecycle status as a protocol error', async () => {
    const client = mockClient({
      create: async () => makeRun({ status: 'exploded' as never }),
    });
    await expect(
      nimbleAgentStartRunTool({ agentId: AGENT_ID, client }).execute!({ task: 't' }, CTX),
    ).rejects.toMatchObject({ reason: 'protocol' });
  });

  it('throws NimbleConfigError without an agent id / api key', async () => {
    const tool = nimbleAgentStartRunTool({});
    vi.stubEnv('NIMBLE_AGENT_ID', '');
    vi.stubEnv('NIMBLE_API_KEY', '');
    await expect(tool.execute!({ task: 't' }, CTX)).rejects.toThrow(NimbleConfigError);
    const toolNoKey = nimbleAgentStartRunTool({ agentId: AGENT_ID });
    await expect(toolNoKey.execute!({ task: 't' }, CTX)).rejects.toThrow(NimbleConfigError);
    vi.unstubAllEnvs();
  });
});

describe('status tool', () => {
  it('maps an active run', async () => {
    const client = mockClient({
      get: async (runId, params) => {
        expect(runId).toBe(RUN_ID);
        expect(params).toEqual({ agent_id: AGENT_ID });
        return makeRun({ status: 'running', started_at: '2026-08-04T10:00:05Z' });
      },
    });
    const out = await nimbleAgentRunStatusTool({ agentId: AGENT_ID, client }).execute!(
      { runId: RUN_ID },
      CTX,
    );
    expect(out).toMatchObject({
      runId: RUN_ID,
      status: 'running',
      isActive: true,
      startedAt: '2026-08-04T10:00:05Z',
    });
  });

  it('reports a failed run without throwing (status is an observation)', async () => {
    const client = mockClient({
      get: async () =>
        makeRun({
          status: 'failed',
          is_active: false,
          error: { message: 'upstream exploded', ref_id: RUN_ID },
        }),
    });
    const out = await nimbleAgentRunStatusTool({
      agentId: AGENT_ID,
      apiKey: TEST_API_KEY,
      client,
    }).execute!(
      { runId: RUN_ID },
      CTX,
    );
    expect(out).toMatchObject({ status: 'failed', error: { message: 'upstream exploded' } });
  });

  it.each([
    ['run id', { id: 'task_run_wrong' }],
    ['agent id', { web_search_agent_id: 'wsa_wrong' }],
  ])('fails closed on a mismatched returned %s', async (_label, patch) => {
    const client = mockClient({ get: async () => makeRun(patch) });
    await expect(
      nimbleAgentRunStatusTool({ agentId: AGENT_ID, client }).execute!(
        { runId: RUN_ID },
        CTX,
      ),
    ).rejects.toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
  });

  it('maps a null status response to a protocol error with requested ids', async () => {
    const client = mockClient({ get: async () => null as never });
    await expect(
      nimbleAgentRunStatusTool({ agentId: AGENT_ID, client }).execute!(
        { runId: RUN_ID },
        CTX,
      ),
    ).rejects.toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
  });

  it('cancels a standalone status read when an injected client never settles', async () => {
    const controller = new AbortController();
    const client = mockClient({ get: async () => await new Promise(() => undefined) });
    const pending = nimbleAgentRunStatusTool({ agentId: AGENT_ID, client }).execute!(
      { runId: RUN_ID },
      { abortSignal: controller.signal } as never,
    );
    controller.abort(new Error('caller cancelled'));
    await expect(pending).rejects.toMatchObject({ reason: 'request', runId: RUN_ID });
  });

  it('does not invoke standalone status get for a pre-aborted caller', async () => {
    const controller = new AbortController();
    controller.abort(new Error('caller cancelled'));
    let calls = 0;
    const client = mockClient({ get: async () => { calls += 1; return makeRun(); } });
    await expect(nimbleAgentRunStatusTool({ agentId: AGENT_ID, client }).execute!(
      { runId: RUN_ID }, { abortSignal: controller.signal } as never,
    )).rejects.toMatchObject({ reason: 'request' });
    expect(calls).toBe(0);
  });

  it('prioritizes cancellation triggered while snapshotting standalone status', async () => {
    const controller = new AbortController();
    const run = new Proxy(makeRun({ status: 'running' }), {
      ownKeys(target) {
        controller.abort(new Error('caller cancelled'));
        return Reflect.ownKeys(target);
      },
    });
    await expect(nimbleAgentRunStatusTool({
      agentId: AGENT_ID, client: mockClient({ get: async () => run }),
    }).execute!({ runId: RUN_ID }, { abortSignal: controller.signal } as never))
      .rejects.toMatchObject({ reason: 'request' });
  });
});

describe('result tool', () => {
  const cfg = { agentId: AGENT_ID, apiKey: TEST_API_KEY };

  it('does not invoke initial status get for a pre-aborted caller', async () => {
    const controller = new AbortController();
    controller.abort(new Error('caller cancelled'));
    let calls = 0;
    const client = mockClient({ get: async () => { calls += 1; return makeRun(); } });
    await expect(nimbleAgentRunResultTool({ ...cfg, client }).execute!(
      { runId: RUN_ID }, { abortSignal: controller.signal } as never,
    )).rejects.toMatchObject({ reason: 'request' });
    expect(calls).toBe(0);
  });

  it('does not invoke result when status snapshotting aborts the caller', async () => {
    const controller = new AbortController();
    let resultCalls = 0;
    const run = new Proxy(makeRun({ status: 'completed', is_active: false }), {
      ownKeys(target) {
        controller.abort(new Error('caller cancelled'));
        return Reflect.ownKeys(target);
      },
    });
    const client = mockClient({
      get: async () => run,
      result: async () => { resultCalls += 1; return makeTextResult(); },
    });
    await expect(nimbleAgentRunResultTool({ ...cfg, client }).execute!(
      { runId: RUN_ID }, { abortSignal: controller.signal } as never,
    )).rejects.toMatchObject({ reason: 'request' });
    expect(resultCalls).toBe(0);
  });

  it('prioritizes cancellation triggered while snapshotting a successful result', async () => {
    const controller = new AbortController();
    const result = new Proxy(makeTextResult(), {
      ownKeys(target) {
        controller.abort(new Error('caller cancelled'));
        return Reflect.ownKeys(target);
      },
    });
    await expect(nimbleAgentRunResultTool({ ...cfg, client: mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => result,
    }) }).execute!({ runId: RUN_ID }, { abortSignal: controller.signal } as never))
      .rejects.toMatchObject({ reason: 'request' });
  });

  it('returns completed/not-ready when result snapshotting consumes the deadline', async () => {
    const result = new Proxy(makeTextResult(), {
      ownKeys(target) {
        const deadline = performance.now() + 30;
        while (performance.now() < deadline) { /* deliberately block */ }
        return Reflect.ownKeys(target);
      },
    });
    await expect(nimbleAgentRunResultTool({
      ...cfg,
      client: mockClient({
        get: async () => makeRun({ status: 'completed', is_active: false }),
        result: async () => result,
      }),
      wait: { timeoutMs: 10, pollIntervalMs: 100 },
    }).execute!({ runId: RUN_ID }, CTX)).resolves.toMatchObject({
      ready: false, status: 'completed', isActive: false,
    });
  });

  it('returns completed/not-ready when slow invalid trust validation crosses the deadline', async () => {
    const result = makeTextResult();
    result.output.trust = structuredClone(result.output.trust);
    result.output.trust.claims.push({ invalid: true } as never);
    const original = nimbleAgentTrustSchema.safeParse.bind(nimbleAgentTrustSchema);
    const parse = vi.spyOn(nimbleAgentTrustSchema, 'safeParse').mockImplementation((value) => {
      const deadline = performance.now() + 30;
      while (performance.now() < deadline) { /* deliberately block */ }
      return original(value);
    });
    await expect(nimbleAgentRunResultTool({
      ...cfg,
      client: mockClient({
        get: async () => makeRun({ status: 'completed', is_active: false }),
        result: async () => result,
      }),
      wait: { timeoutMs: 10, pollIntervalMs: 100 },
    }).execute!({ runId: RUN_ID }, CTX)).resolves.toMatchObject({
      ready: false, status: 'completed', isActive: false,
    });
    parse.mockRestore();
  });

  it('prioritizes cancellation triggered by result error status inspection', async () => {
    const controller = new AbortController();
    const error = new Error('conflict') as Error & { status?: number };
    Object.defineProperty(error, 'status', {
      get() { controller.abort(new Error('caller cancelled')); return 409; },
    });
    await expect(nimbleAgentRunResultTool({ ...cfg, client: mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => { throw error; },
    }) }).execute!({ runId: RUN_ID }, { abortSignal: controller.signal } as never))
      .rejects.toMatchObject({ reason: 'request' });
  });

  it('honors the deadline crossed during result error status inspection', async () => {
    const error = new Error('conflict') as Error & { status?: number };
    Object.defineProperty(error, 'status', {
      get() {
        const deadline = performance.now() + 30;
        while (performance.now() < deadline) { /* deliberately block */ }
        return 409;
      },
    });
    await expect(nimbleAgentRunResultTool({
      ...cfg,
      client: mockClient({
        get: async () => makeRun({ status: 'completed', is_active: false }),
        result: async () => { throw error; },
      }),
      wait: { timeoutMs: 10, pollIntervalMs: 100 },
    }).execute!({ runId: RUN_ID }, CTX)).resolves.toMatchObject({
      ready: false, status: 'completed', isActive: false,
    });
  });

  it('prioritizes cancellation triggered while parsing a 422 body', async () => {
    const controller = new AbortController();
    const body = new Proxy(makeFailedResult('failed'), {
      ownKeys(target) {
        controller.abort(new Error('caller cancelled'));
        return Reflect.ownKeys(target);
      },
    });
    const error = httpError(422, body);
    await expect(nimbleAgentRunResultTool({ ...cfg, client: mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => { throw error; },
    }) }).execute!({ runId: RUN_ID }, { abortSignal: controller.signal } as never))
      .rejects.toMatchObject({ reason: 'request' });
  });

  it('honors the deadline crossed while parsing a 422 body', async () => {
    const body = new Proxy(makeFailedResult('failed'), {
      ownKeys(target) {
        const deadline = performance.now() + 30;
        while (performance.now() < deadline) { /* deliberately block */ }
        return Reflect.ownKeys(target);
      },
    });
    await expect(nimbleAgentRunResultTool({
      ...cfg,
      client: mockClient({
        get: async () => makeRun({ status: 'completed', is_active: false }),
        result: async () => { throw httpError(422, body); },
      }),
      wait: { timeoutMs: 10, pollIntervalMs: 100 },
    }).execute!({ runId: RUN_ID }, CTX)).resolves.toMatchObject({
      ready: false, status: 'completed', isActive: false,
    });
  });

  it('returns { ready: false } while queued/running — never blocks by default', async () => {
    for (const status of ['queued', 'running'] as const) {
      const client = mockClient({ get: async () => makeRun({ status }) });
      const out = await nimbleAgentRunResultTool({ ...cfg, client }).execute!(
        { runId: RUN_ID },
        CTX,
      );
      expect(out).toMatchObject({ ready: false, status, isActive: true });
    }
  });

  it('returns the completed text answer with verbatim trust', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => makeTextResult(),
    });
    const out = (await nimbleAgentRunResultTool({ ...cfg, client }).execute!(
      { runId: RUN_ID },
      CTX,
    )) as NimbleAgentRunCompletedOutput;
    expect(out.ready).toBe(true);
    expect(out.output).toMatchObject({ type: 'text', text: 'The answer [1].' });
    expect(out.output.trust).toEqual(makeTextResult().output.trust);
  });

  it('returns the completed json answer as structured data', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => makeJsonResult(),
    });
    const out = (await nimbleAgentRunResultTool({ ...cfg, client }).execute!(
      { runId: RUN_ID },
      CTX,
    )) as NimbleAgentRunCompletedOutput;
    expect(out.output).toMatchObject({ type: 'json', json: { rows: [{ name: 'x' }] } });
  });

  it('rejects sparse arrays instead of silently changing their length or holes', async () => {
    const sparse = [1] as unknown[];
    sparse.length = 3;
    const result = makeJsonResult();
    result.output.content = sparse;
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => result,
    });
    await expect(
      nimbleAgentRunResultTool({ ...cfg, client }).execute!({ runId: RUN_ID }, CTX),
    ).rejects.toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
  });

  it('uses one validated array length descriptor when snapshotting proxy results', async () => {
    let lengthReads = 0;
    const content = new Proxy([1], {
      getOwnPropertyDescriptor(target, key) {
        const descriptor = Reflect.getOwnPropertyDescriptor(target, key);
        if (key !== 'length' || !descriptor) return descriptor;
        lengthReads += 1;
        return { ...descriptor, value: lengthReads === 1 ? 1 : 2 };
      },
    });
    const result = makeJsonResult();
    result.output.content = content;
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => result,
    });
    const out = (await nimbleAgentRunResultTool({ ...cfg, client }).execute!(
      { runId: RUN_ID },
      CTX,
    )) as NimbleAgentRunCompletedOutput;
    expect(lengthReads).toBe(1);
    expect(out.output).toMatchObject({ type: 'json', json: [1] });
  });

  it('rejects array proxies that replace a required index with an out-of-range index', async () => {
    const content = new Proxy([1], {
      ownKeys: () => ['length', '2'],
      getOwnPropertyDescriptor(target, key) {
        if (key === '2') {
          return { value: 1, enumerable: true, configurable: true, writable: true };
        }
        return Reflect.getOwnPropertyDescriptor(target, key);
      },
    });
    const result = makeJsonResult();
    result.output.content = content;
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => result,
    });
    await expect(
      nimbleAgentRunResultTool({ ...cfg, client }).execute!({ runId: RUN_ID }, CTX),
    ).rejects.toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
  });

  it('sanitizes callable proxy errors whose string conversion throws the API key', async () => {
    const callable = new Proxy(() => undefined, {
      get(target, key, receiver) {
        if (key === Symbol.toPrimitive) throw new Error(`conversion ${TEST_API_KEY}`);
        return Reflect.get(target, key, receiver);
      },
    });
    const client = mockClient({ get: async () => { throw callable; } });
    const err = await nimbleAgentRunResultTool({ ...cfg, client })
      .execute!({ runId: RUN_ID }, CTX)
      .catch((caught: unknown) => caught);
    expect(err).toBeInstanceOf(NimbleAgentRunError);
    expect(String(err)).not.toContain(TEST_API_KEY);
    expect(inspect(err, { depth: 5 })).not.toContain(TEST_API_KEY);
  });

  it('throws with runId preserved when the run failed / was cancelled', async () => {
    for (const status of ['failed', 'cancelled'] as const) {
      const client = mockClient({
        get: async () => makeRun({ status, is_active: false }),
      });
      const err = await nimbleAgentRunResultTool({ ...cfg, client })
        .execute!({ runId: RUN_ID }, CTX)
        .then(
          () => {
            throw new Error('expected failure');
          },
          (e: unknown) => e as NimbleAgentRunError,
        );
      expect(err).toBeInstanceOf(NimbleAgentRunError);
      expect(err.reason).toBe(status === 'failed' ? 'failed' : 'cancelled');
      expect(err.runId).toBe(RUN_ID);
      expect(err.message).toContain(RUN_ID);
    }
  });

  it.each(['failed', 'cancelled'] as const)(
    'bounds and scrubs an oversized %s terminal detail before formatting',
    async (status) => {
      const oversized = `${'x'.repeat(4_090)}${TEST_API_KEY}${'y'.repeat(2_000_000)}`;
      const err = await nimbleAgentRunResultTool({
        ...cfg,
        client: mockClient({
          get: async () => makeRun({
            status,
            is_active: false,
            error: { message: oversized, ref_id: RUN_ID },
          }),
        }),
      }).execute!({ runId: RUN_ID }, CTX).catch(
        (caught: unknown) => caught as NimbleAgentRunError,
      ) as NimbleAgentRunError;
      expect(err).toBeInstanceOf(NimbleAgentRunError);
      expect(err.message).toContain('[redacted]');
      expect(err.message).toContain('[truncated]');
      expect(err.message).not.toContain(TEST_API_KEY);
      expect(err.message.length).toBeLessThan(4_300);
    },
  );

  it.each([
    ['before', 4_095, true],
    ['at', 4_096, false],
    ['after', 4_097, false],
  ] as const)('keeps credential look-ahead private when a key starts %s the cap', async (_, offset, redacted) => {
    const detail = `${'x'.repeat(offset)}${TEST_API_KEY}${'y'.repeat(8_192)}`;
    const err = await nimbleAgentRunResultTool({
      ...cfg,
      client: mockClient({
        get: async () => makeRun({
          status: 'failed',
          is_active: false,
          error: { message: detail, ref_id: RUN_ID },
        }),
      }),
    }).execute!({ runId: RUN_ID }, CTX).catch(
      (caught: unknown) => caught as NimbleAgentRunError,
    ) as NimbleAgentRunError;
    expect(err.message).not.toContain(TEST_API_KEY);
    expect(err.message.includes('[redacted]')).toBe(redacted);
    expect(err.message.length).toBeLessThan(4_300);
  });

  it('omits oversized terminal detail when the configured credential itself exceeds the cap', async () => {
    const longKey = `key-${'s'.repeat(5_000)}`;
    const err = await nimbleAgentRunResultTool({
      agentId: AGENT_ID,
      apiKey: longKey,
      client: mockClient({
        get: async () => makeRun({
          status: 'failed',
          is_active: false,
          error: { message: `${'x'.repeat(4_000)}${longKey}`, ref_id: RUN_ID },
        }),
      }),
    }).execute!({ runId: RUN_ID }, CTX).catch(
      (caught: unknown) => caught as NimbleAgentRunError,
    ) as NimbleAgentRunError;
    expect(err.message).toContain('[server detail omitted: exceeded safe display limit]');
    expect(err.message).not.toContain(longKey.slice(0, 128));
    expect(err.message.length).toBeLessThan(300);
  });

  it.each(['failed', 'cancelled'] as const)(
    'returns authoritative %s/not-ready when large terminal error formatting reaches the deadline',
    async (status) => {
      let reads = 0;
      const clock = vi.spyOn(performance, 'now').mockImplementation(() => {
        reads += 1;
        return reads < 3 ? 0 : 20;
      });
      try {
        const out = await nimbleAgentRunResultTool({
          ...cfg,
          client: mockClient({
            get: async () => makeRun({
              status,
              is_active: false,
              error: { message: 'large terminal detail '.repeat(2_000), ref_id: RUN_ID },
            }),
          }),
          wait: { timeoutMs: 10, pollIntervalMs: 100 },
        }).execute!({ runId: RUN_ID }, CTX);
        expect(out).toMatchObject({ ready: false, status, isActive: false, runId: RUN_ID });
      } finally {
        clock.mockRestore();
      }
    },
  );

  it('maps a 409 from the result endpoint to { ready: false }', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => {
        throw httpError(409, { detail: 'still active' });
      },
    });
    const out = await nimbleAgentRunResultTool({ ...cfg, client }).execute!(
      { runId: RUN_ID },
      CTX,
    );
    expect(out).toMatchObject({ ready: false, status: 'running' });
  });

  it('retries a transient result 409 within the bounded wait', async () => {
    let resultCalls = 0;
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => {
        resultCalls += 1;
        if (resultCalls === 1) throw httpError(409, { detail: 'still active' });
        return makeTextResult();
      },
    });
    const out = await nimbleAgentRunResultTool({
      ...cfg,
      client,
      wait: { timeoutMs: 250, pollIntervalMs: 100 },
    }).execute!({ runId: RUN_ID }, CTX);
    expect(out).toMatchObject({ ready: true, status: 'completed' });
    expect(resultCalls).toBe(2);
  });

  it('stops retrying result 409 at the original wait deadline', async () => {
    let resultCalls = 0;
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => {
        resultCalls += 1;
        throw httpError(409, { detail: 'still active' });
      },
    });
    const started = performance.now();
    const out = await nimbleAgentRunResultTool({
      ...cfg,
      client,
      wait: { timeoutMs: 150, pollIntervalMs: 100 },
    }).execute!({ runId: RUN_ID }, CTX);
    expect(out).toMatchObject({ ready: false, status: 'completed', isActive: false });
    expect(resultCalls).toBeGreaterThanOrEqual(2);
    expect(performance.now() - started).toBeLessThan(500);
  });

  it('does not invoke result again when the retry sleep consumes the deadline', async () => {
    let resultCalls = 0;
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => {
        resultCalls += 1;
        throw httpError(409, { detail: 'still active' });
      },
    });
    await expect(
      nimbleAgentRunResultTool({
        ...cfg,
        client,
        wait: { timeoutMs: 50, pollIntervalMs: 100 },
      }).execute!({ runId: RUN_ID }, CTX),
    ).resolves.toMatchObject({ ready: false, status: 'completed' });
    expect(resultCalls).toBe(1);
  });

  it('maps a 422 with a bare failed body to a terminal failure with the server message', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => {
        throw httpError(422, makeFailedResult('server says no'));
      },
    });
    await expect(
      nimbleAgentRunResultTool({ ...cfg, client }).execute!({ runId: RUN_ID }, CTX),
    ).rejects.toMatchObject({ reason: 'failed', runId: RUN_ID, message: expect.stringContaining('server says no') });
  });

  it('maps a 422 with a gateway-wrapped { detail } body the same way', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => {
        throw httpError(422, { detail: makeFailedResult('wrapped failure') });
      },
    });
    await expect(
      nimbleAgentRunResultTool({ ...cfg, client }).execute!({ runId: RUN_ID }, CTX),
    ).rejects.toMatchObject({ reason: 'failed', message: expect.stringContaining('wrapped failure') });
  });

  it.each(['failed', 'cancelled'] as const)(
    'retains the prior completed snapshot when slow valid 422 %s identity validation reaches the deadline',
    async (status) => {
      let reads = 0;
      const clock = vi.spyOn(performance, 'now').mockImplementation(() => {
        reads += 1;
        return reads < 6 ? 0 : 20;
      });
      const failed = {
        ...makeFailedResult('late terminal failure'),
        run: makeRun({
          status,
          is_active: false,
          future_graph: Array.from({ length: 128 }, (_, index) => ({ index, value: `safe-${index}` })),
        } as never),
      };
      try {
        const out = await nimbleAgentRunResultTool({
          ...cfg,
          client: mockClient({
            get: async () => makeRun({ status: 'completed', is_active: false }),
            result: async () => { throw httpError(422, failed); },
          }),
          wait: { timeoutMs: 10, pollIntervalMs: 100 },
        }).execute!({ runId: RUN_ID }, CTX);
        expect(out).toMatchObject({
          ready: false, status: 'completed', runId: RUN_ID, agentId: AGENT_ID,
        });
      } finally {
        clock.mockRestore();
      }
    },
  );

  it('retains the prior completed snapshot when malformed 422 identity validation reaches the deadline', async () => {
    let reads = 0;
    const clock = vi.spyOn(performance, 'now').mockImplementation(() => {
      reads += 1;
      return reads < 6 ? 0 : 20;
    });
    const failed = {
      ...makeFailedResult('foreign failure'),
      run: makeRun({
        status: 'failed',
        is_active: false,
        id: 'task_run_ffffffff-ffff-ffff-ffff-ffffffffffff',
        future_graph: Array.from({ length: 128 }, (_, index) => ({ index })),
      } as never),
    };
    try {
      const out = await nimbleAgentRunResultTool({
        ...cfg,
        client: mockClient({
          get: async () => makeRun({ status: 'completed', is_active: false }),
          result: async () => { throw httpError(422, failed); },
        }),
        wait: { timeoutMs: 10, pollIntervalMs: 100 },
      }).execute!({ runId: RUN_ID }, CTX);
      expect(out).toMatchObject({ ready: false, status: 'completed', runId: RUN_ID });
      expect(JSON.stringify(out)).not.toContain('ffffffff');
    } finally {
      clock.mockRestore();
    }
  });

  it.each([undefined, 42])(
    'rejects a 422 failed envelope with invalid error.ref_id %s',
    async (refId) => {
      const failed = makeFailedResult('failed');
      Object.assign(failed.error, { ref_id: refId });
      const client = mockClient({
        get: async () => makeRun({ status: 'completed', is_active: false }),
        result: async () => { throw httpError(422, failed); },
      });
      await expect(
        nimbleAgentRunResultTool({ ...cfg, client }).execute!({ runId: RUN_ID }, CTX),
      ).rejects.toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
    },
  );

  it.each(['credential', 'active', 'failed'] as const)(
    'retains the prior completed snapshot when a large %s envelope scan reaches the deadline',
    async (kind) => {
      let reads = 0;
      const clock = vi.spyOn(performance, 'now').mockImplementation(() => {
        reads += 1;
        return reads < 5 ? 0 : 20;
      });
      const padding = Array.from({ length: 128 }, (_, index) => `safe-${index}`);
      const result = kind === 'failed'
        ? { ...makeFailedResult('foreign failure'), padding }
        : {
            ...makeTextResult(),
            ...(kind === 'active'
              ? { run: makeRun({ status: 'running', is_active: true }) }
              : {}),
            padding: kind === 'credential' ? [...padding, FAKE_KEY] : padding,
          };
      try {
        const out = await nimbleAgentRunResultTool({
          ...cfg,
          client: mockClient({
            get: async () => makeRun({ status: 'completed', is_active: false }),
            result: async () => result,
          }),
          wait: { timeoutMs: 10, pollIntervalMs: 100 },
        }).execute!({ runId: RUN_ID }, CTX);
        expect(out).toMatchObject({
          ready: false, status: 'completed', runId: RUN_ID, agentId: AGENT_ID,
        });
        expect(JSON.stringify(out)).not.toContain(FAKE_KEY);
      } finally {
        clock.mockRestore();
      }
    },
  );

  it.each(['error', 'detail'] as const)(
    'sanitizes a 422 with a throwing %s accessor',
    async (property) => {
      const err = httpError(422);
      const target = property === 'error' ? err : {};
      if (property === 'detail') err.error = target;
      Object.defineProperty(target, property, {
        get() { throw new Error(`${property} exposed ${FAKE_KEY}`); },
      });
      const client = mockClient({
        get: async () => makeRun({ status: 'completed', is_active: false }),
        result: async () => { throw err; },
      });
      const mapped = await nimbleAgentRunResultTool({ ...cfg, client })
        .execute!({ runId: RUN_ID }, CTX)
        .then(
          () => { throw new Error('expected failure'); },
          (error: unknown) => error as NimbleAgentRunError,
        );
      expect(mapped.message).not.toContain(FAKE_KEY);
      expect(inspect(mapped, { depth: 20 })).not.toContain(FAKE_KEY);
    },
  );

  it('rejects a 422 envelope with a stateful error.message accessor', async () => {
    const failed = makeFailedResult('first read');
    let reads = 0;
    Object.defineProperty(failed.error, 'message', {
      get() {
        reads += 1;
        if (reads === 1) return 'first read';
        throw new Error(`second read exposed ${FAKE_KEY}`);
      },
    });
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => { throw httpError(422, failed); },
    });
    const err = await nimbleAgentRunResultTool({ ...cfg, client })
      .execute!({ runId: RUN_ID }, CTX)
      .then(
        () => { throw new Error('expected failure'); },
        (error: unknown) => error as NimbleAgentRunError,
      );
    expect(reads).toBe(0);
    expect(err.message).not.toContain(FAKE_KEY);
    expect(inspect(err, { depth: 20 })).not.toContain(FAKE_KEY);
  });

  it.each(['queued', 'running', 'completed'] as const)('rejects a 422 envelope with non-failure status %s', async (status) => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => {
        throw httpError(422, {
          ...makeFailedResult('contradictory envelope'),
          run: makeRun({ status, is_active: status === 'queued' || status === 'running' }),
        });
      },
    });
    await expect(
      nimbleAgentRunResultTool({ ...cfg, client }).execute!({ runId: RUN_ID }, CTX),
    ).rejects.toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
  });

  it('re-validates the run embedded in the result payload (eventual consistency)', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => ({
        run: makeRun({ status: 'running' }),
        output: makeTextResult().output,
      }),
    });
    const out = await nimbleAgentRunResultTool({ ...cfg, client }).execute!(
      { runId: RUN_ID },
      CTX,
    );
    expect(out).toMatchObject({ ready: false, status: 'running' });
  });

  it.each(['failed', 'cancelled', 'malformed'] as const)(
    'retains the prior completed snapshot when slow embedded %s run validation reaches the deadline',
    async (kind) => {
      let reads = 0;
      const clock = vi.spyOn(performance, 'now').mockImplementation(() => {
        reads += 1;
        return reads < 6 ? 0 : 20;
      });
      try {
        const out = await nimbleAgentRunResultTool({
          ...cfg,
          client: mockClient({
            get: async () => makeRun({ status: 'completed', is_active: false }),
            result: async () => ({
              ...makeTextResult(),
              run: makeRun({
                status: kind === 'malformed' ? ('unexpected' as never) : kind,
                is_active: false,
              }),
            }),
          }),
          wait: { timeoutMs: 10, pollIntervalMs: 100 },
        }).execute!({ runId: RUN_ID }, CTX);
        expect(out).toMatchObject({
          ready: false, status: 'completed', runId: RUN_ID, agentId: AGENT_ID,
        });
      } finally {
        clock.mockRestore();
      }
    },
  );

  it('retries an active successful result envelope within the bounded wait', async () => {
    let resultCalls = 0;
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => {
        resultCalls += 1;
        if (resultCalls === 1) {
          return { run: makeRun({ status: 'running', is_active: true }), output: makeTextResult().output };
        }
        return makeTextResult();
      },
    });
    const out = await nimbleAgentRunResultTool({
      ...cfg,
      client,
      wait: { timeoutMs: 250, pollIntervalMs: 100 },
    }).execute!({ runId: RUN_ID }, CTX);
    expect(out).toMatchObject({ ready: true, status: 'completed' });
    expect(resultCalls).toBe(2);
  });

  it('stops retrying active successful result envelopes at the original deadline', async () => {
    let resultCalls = 0;
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => {
        resultCalls += 1;
        return { run: makeRun({ status: 'running', is_active: true }), output: makeTextResult().output };
      },
    });
    const out = await nimbleAgentRunResultTool({
      ...cfg,
      client,
      wait: { timeoutMs: 50, pollIntervalMs: 100 },
    }).execute!({ runId: RUN_ID }, CTX);
    expect(out).toMatchObject({ ready: false, status: 'completed', isActive: false });
    expect(resultCalls).toBe(1);
  });

  it.each([
    ['run id', { id: 'task_run_wrong' }],
    ['agent id', { web_search_agent_id: 'wsa_wrong' }],
  ])('fails closed when a completed result returns a mismatched %s', async (_label, patch) => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => ({
        ...makeTextResult(),
        run: makeRun({ status: 'completed', is_active: false, ...patch }),
      }),
    });
    await expect(
      nimbleAgentRunResultTool({ ...cfg, client }).execute!({ runId: RUN_ID }, CTX),
    ).rejects.toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
  });

  it('fails closed when a 422 failure body returns mismatched ids', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => {
        throw httpError(422, {
          ...makeFailedResult('wrong run'),
          run: makeRun({ id: 'task_run_wrong', status: 'failed', is_active: false }),
        });
      },
    });
    await expect(
      nimbleAgentRunResultTool({ ...cfg, client }).execute!({ runId: RUN_ID }, CTX),
    ).rejects.toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
  });

  it('maps malformed result payloads to protocol errors, not TypeErrors', async () => {
    const cases: unknown[] = [
      null,
      {},
      { run: makeRun({ status: 'completed' }) },
      { run: makeRun({ status: 'completed' }), error: { message: 'not a failure' } },
      { run: makeRun({ status: 'failed' }), error: {} },
      { run: makeRun({ status: 'completed' }), output: { content: 42 } },
      { run: makeRun({ status: 'completed' }), output: { type: 'text', content: 'x' } }, // no trust
      { run: makeRun({ status: 'completed' }), output: { type: 'text', content: 'x', trust: {} } },
      {
        run: makeRun({ status: 'completed' }),
        output: {
          type: 'text',
          content: 'x',
          trust: { ...makeTextResult().output.trust, confidence: 'certain' },
        },
      },
    ];
    for (const payload of cases) {
      const client = mockClient({
        get: async () => makeRun({ status: 'completed', is_active: false }),
        result: async () => payload as never,
      });
      await expect(
        nimbleAgentRunResultTool({ ...cfg, client }).execute!({ runId: RUN_ID }, CTX),
      ).rejects.toMatchObject({ reason: 'protocol' });
    }
  });

  it.each([
    { effort: undefined },
    { created_at: undefined },
    { is_active: 'yes' },
    { error: { message: 42, ref_id: RUN_ID } },
  ])('maps malformed read run field %# to a protocol error', async (patch) => {
    const client = mockClient({
      get: async () => makeRun(patch as never),
    });
    await expect(
      nimbleAgentRunStatusTool({ ...cfg, client }).execute!({ runId: RUN_ID }, CTX),
    ).rejects.toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
  });

  it.each([
    ['queued', false],
    ['running', false],
    ['completed', true],
    ['failed', true],
    ['cancelled', true],
  ] as const)(
    'rejects contradictory lifecycle metadata (%s, is_active=%s)',
    async (status, isActive) => {
      const client = mockClient({
        get: async () => makeRun({ status, is_active: isActive }),
      });
      await expect(
        nimbleAgentRunStatusTool({ ...cfg, client }).execute!({ runId: RUN_ID }, CTX),
      ).rejects.toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
    },
  );

  it('rejects and redacts an API key reflected in read metadata', async () => {
    const client = mockClient({
      get: async () => makeRun({ created_at: `time-${FAKE_KEY}` }),
    });
    const err = await nimbleAgentRunStatusTool({ agentId: AGENT_ID, apiKey: FAKE_KEY, client })
      .execute!({ runId: RUN_ID }, CTX)
      .then(
        () => { throw new Error('expected failure'); },
        (error: unknown) => error as NimbleAgentRunError,
      );
    expect(err).toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
    expect(err.message).not.toContain(FAKE_KEY);
    expect(JSON.stringify(err)).not.toContain(FAKE_KEY);
  });

  it('rejects and redacts an API key reflected in result run metadata', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => ({
        ...makeTextResult(),
        run: makeRun({
          status: 'completed',
          is_active: false,
          created_at: `time-${FAKE_KEY}`,
        }),
      }),
    });
    const err = await nimbleAgentRunResultTool({ agentId: AGENT_ID, apiKey: FAKE_KEY, client })
      .execute!({ runId: RUN_ID }, CTX)
      .then(
        () => { throw new Error('expected failure'); },
        (error: unknown) => error as NimbleAgentRunError,
      );
    expect(err).toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
    expect(err.message).not.toContain(FAKE_KEY);
    expect(JSON.stringify(err)).not.toContain(FAKE_KEY);
  });

  it('does not expose an unknown read status in a protocol error', async () => {
    const leakyStatus = `status-${FAKE_KEY}`;
    const client = mockClient({
      get: async () => makeRun({ status: leakyStatus as never }),
    });
    const err = await nimbleAgentRunStatusTool({ ...cfg, client })
      .execute!({ runId: RUN_ID }, CTX)
      .then(
        () => { throw new Error('expected failure'); },
        (error: unknown) => error as NimbleAgentRunError,
      );
    expect(err.runStatus).toBeUndefined();
    expect(err.message).not.toContain(leakyStatus);
    expect(JSON.stringify(err)).not.toContain(FAKE_KEY);
  });

  it('result body in the failed form ({ run, error }) throws the terminal failure', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => makeFailedResult('late failure'),
    });
    await expect(
      nimbleAgentRunResultTool({ ...cfg, client }).execute!({ runId: RUN_ID }, CTX),
    ).rejects.toMatchObject({ reason: 'failed', message: expect.stringContaining('late failure') });
  });

  it.each(['failed', 'cancelled'] as const)(
    'retains the prior completed snapshot when %s-envelope validation crosses the deadline',
    async (status) => {
      let reads = 0;
      const clock = vi.spyOn(performance, 'now').mockImplementation(() => {
        reads += 1;
        return reads < 5 ? 0 : 20;
      });
      try {
        const out = await nimbleAgentRunResultTool({
          ...cfg,
          client: mockClient({
            get: async () => makeRun({ status: 'completed', is_active: false }),
            result: async () => ({
              ...makeFailedResult('late failure'),
              run: makeRun({ status, is_active: false }),
            }),
          }),
          wait: { timeoutMs: 10, pollIntervalMs: 100 },
        }).execute!({ runId: RUN_ID }, CTX);
        expect(out).toMatchObject({
          ready: false, status: 'completed', isActive: false, runId: RUN_ID, agentId: AGENT_ID,
        });
      } finally {
        clock.mockRestore();
      }
    },
  );

  it.each([
    ['run id', { id: 'task_run_ffffffff-ffff-ffff-ffff-ffffffffffff' }],
    ['agent id', { web_search_agent_id: 'wsa_ffffffff-ffff-4fff-8fff-ffffffffffff' }],
  ] as const)(
    'does not expose a mismatched %s when failure-envelope validation reaches the deadline',
    async (_label, patch) => {
      let reads = 0;
      const clock = vi.spyOn(performance, 'now').mockImplementation(() => {
        reads += 1;
        return reads < 5 ? 0 : 20;
      });
      try {
        const out = await nimbleAgentRunResultTool({
          ...cfg,
          client: mockClient({
            get: async () => makeRun({ status: 'completed', is_active: false }),
            result: async () => ({
              ...makeFailedResult('foreign failure'),
              run: makeRun({ status: 'failed', is_active: false, ...patch }),
            }),
          }),
          wait: { timeoutMs: 10, pollIntervalMs: 100 },
        }).execute!({ runId: RUN_ID }, CTX);
        expect(out).toMatchObject({
          ready: false, status: 'completed', runId: RUN_ID, agentId: AGENT_ID,
        });
        expect(JSON.stringify(out)).not.toContain('ffffffff');
      } finally {
        clock.mockRestore();
      }
    },
  );

  it.each([undefined, 42])(
    'rejects a returned failed envelope with invalid error.ref_id %s',
    async (refId) => {
      const failed = makeFailedResult('failed');
      Object.assign(failed.error, { ref_id: refId });
      const client = mockClient({
        get: async () => makeRun({ status: 'completed', is_active: false }),
        result: async () => failed,
      });
      await expect(
        nimbleAgentRunResultTool({ ...cfg, client }).execute!({ runId: RUN_ID }, CTX),
      ).rejects.toMatchObject({ reason: 'protocol', runId: RUN_ID, agentId: AGENT_ID });
    },
  );

  it('opt-in wait polls until completion and honors the interval floor', async () => {
    let calls = 0;
    const client = mockClient({
      get: async () => {
        calls += 1;
        return calls < 3
          ? makeRun({ status: 'running' })
          : makeRun({ status: 'completed', is_active: false });
      },
      result: async () => makeTextResult(),
    });
    const out = await nimbleAgentRunResultTool({
      ...cfg,
      client,
      wait: { timeoutMs: 5_000, pollIntervalMs: 1 }, // floored to 100ms
    }).execute!({ runId: RUN_ID }, CTX);
    expect(calls).toBe(3);
    expect(out).toMatchObject({ ready: true, status: 'completed' });
  });

  it('a wait that times out returns { ready: false } — a live run is not an error', async () => {
    let calls = 0;
    const client = mockClient({ get: async () => { calls += 1; return makeRun({ status: 'running' }); } });
    const out = await nimbleAgentRunResultTool({
      ...cfg,
      client,
      wait: { timeoutMs: 150, pollIntervalMs: 1 },
    }).execute!({ runId: RUN_ID }, CTX);
    expect(out).toMatchObject({ ready: false, status: 'running' });
    expect(calls).toBe(2);
  });

  it('normalizes fractional wait durations before constructing timeout signals', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => makeTextResult(),
    });
    await expect(
      nimbleAgentRunResultTool({
        ...cfg,
        client,
        wait: { timeoutMs: 150.5, pollIntervalMs: 100.5 },
      }).execute!({ runId: RUN_ID }, CTX),
    ).resolves.toMatchObject({ ready: true, status: 'completed' });
  });

  it.each([
    ['regular poll sleep', { timeoutMs: 5_000, pollIntervalMs: 100 }],
    ['final remaining-time sleep', { timeoutMs: 50, pollIntervalMs: 100 }],
  ])('sanitizes credential-bearing abort reasons from the %s', async (_name, wait) => {
    const controller = new AbortController();
    const client = mockClient({ get: async () => makeRun({ status: 'running' }) });
    const pending = nimbleAgentRunResultTool({ ...cfg, client, wait }).execute!(
      { runId: RUN_ID },
      { abortSignal: controller.signal } as never,
    );
    setTimeout(() => controller.abort(new Error(`cancelled ${TEST_API_KEY}`)), 10);

    const err = await pending.catch((caught: unknown) => caught);
    expect(err).toBeInstanceOf(NimbleAgentRunError);
    expect(err).toMatchObject({ runId: RUN_ID, agentId: AGENT_ID });
    expect(String(err)).not.toContain(TEST_API_KEY);
    expect(inspect(err, { depth: 5 })).not.toContain(TEST_API_KEY);
  });

  it('bounds an in-flight status poll by the remaining wait timeout', async () => {
    let calls = 0;
    const client = mockClient({
      get: async (_runId, _query, options) => {
        calls += 1;
        if (calls === 1) return makeRun({ status: 'running' });
        return await new Promise((_, reject) => {
          options?.signal?.addEventListener('abort', () => reject(options.signal?.reason), {
            once: true,
          });
        });
      },
    });
    const started = performance.now();
    const out = await nimbleAgentRunResultTool({
      ...cfg,
      client,
      wait: { timeoutMs: 175, pollIntervalMs: 100 },
    }).execute!({ runId: RUN_ID }, CTX);
    expect(out).toMatchObject({ ready: false, status: 'running' });
    expect(calls).toBe(2);
    expect(performance.now() - started).toBeLessThan(500);
  });

  it('bounds the initial status request by the wait timeout', async () => {
    const client = mockClient({
      get: async (_runId, _query, options) =>
        await new Promise((_, reject) => {
          options?.signal?.addEventListener('abort', () => reject(options.signal?.reason), {
            once: true,
          });
        }),
    });
    const started = performance.now();
    const out = await nimbleAgentRunResultTool({
      ...cfg,
      client,
      wait: { timeoutMs: 150, pollIntervalMs: 100 },
    }).execute!({ runId: RUN_ID }, CTX);
    expect(out).toMatchObject({
      ready: false,
      runId: RUN_ID,
      agentId: AGENT_ID,
      status: 'unknown',
    });
    expect(out).not.toHaveProperty('isActive');
    expect(performance.now() - started).toBeLessThan(500);
  });

  it('bounds a noncooperative initial status request', async () => {
    const client = mockClient({
      get: async () => await new Promise<never>(() => {}),
    });
    await expect(
      nimbleAgentRunResultTool({
        ...cfg,
        client,
        wait: { timeoutMs: 20, pollIntervalMs: 10 },
      }).execute!({ runId: RUN_ID }, CTX),
    ).resolves.toEqual({
      ready: false,
      runId: RUN_ID,
      agentId: AGENT_ID,
      status: 'unknown',
    });
  });

  it('rejects a synchronous late status success before result retrieval', async () => {
    let resultCalls = 0;
    const client = mockClient({
      get: async () => {
        const deadline = performance.now() + 30;
        while (performance.now() < deadline) { /* deliberately block */ }
        return makeRun({ status: 'completed', is_active: false });
      },
      result: async () => {
        resultCalls += 1;
        return makeTextResult();
      },
    });
    await expect(
      nimbleAgentRunResultTool({
        ...cfg,
        client,
        wait: { timeoutMs: 10, pollIntervalMs: 5 },
      }).execute!({ runId: RUN_ID }, CTX),
    ).resolves.toMatchObject({ ready: false, status: 'completed', isActive: false });
    expect(resultCalls).toBe(0);
  });

  it.each(['failed', 'cancelled'] as const)(
    'maps a synchronous late initial %s snapshot to terminal/not-ready',
    async (status) => {
      const client = mockClient({
        get: async () => {
          const deadline = performance.now() + 30;
          while (performance.now() < deadline) { /* deliberately block */ }
          return makeRun({ status, is_active: false });
        },
      });
      await expect(
        nimbleAgentRunResultTool({
          ...cfg,
          client,
          wait: { timeoutMs: 10, pollIntervalMs: 5 },
        }).execute!({ runId: RUN_ID }, CTX),
      ).resolves.toMatchObject({ ready: false, status, isActive: false });
    },
  );

  it('prioritizes caller cancellation over a late terminal poll snapshot', async () => {
    const controller = new AbortController();
    let calls = 0;
    const client = mockClient({
      get: async () => {
        calls += 1;
        if (calls === 1) return makeRun({ status: 'running', is_active: true });
        controller.abort(new Error('caller cancelled'));
        const deadline = performance.now() + 300;
        while (performance.now() < deadline) { /* deliberately block */ }
        return makeRun({ status: 'failed', is_active: false });
      },
    });
    const pending = nimbleAgentRunResultTool({
      ...cfg,
      client,
      wait: { timeoutMs: 250, pollIntervalMs: 10 },
    }).execute!({ runId: RUN_ID }, { abortSignal: controller.signal } as never);
    await expect(pending).rejects.toBeInstanceOf(NimbleAgentRunError);
    expect(calls).toBe(2);
  });

  it('maps a synchronous late initial-status rejection to unknown', async () => {
    const client = mockClient({
      get: async () => {
        const deadline = performance.now() + 30;
        while (performance.now() < deadline) { /* deliberately block */ }
        throw new Error('late status failure');
      },
    });
    await expect(
      nimbleAgentRunResultTool({
        ...cfg,
        client,
        wait: { timeoutMs: 10, pollIntervalMs: 5 },
      }).execute!({ runId: RUN_ID }, CTX),
    ).resolves.toEqual({
      ready: false,
      runId: RUN_ID,
      agentId: AGENT_ID,
      status: 'unknown',
    });
  });

  it('bounds a noncooperative later status poll', async () => {
    let calls = 0;
    const client = mockClient({
      get: async () => {
        calls += 1;
        if (calls === 1) return makeRun({ status: 'running', is_active: true });
        return await new Promise<never>(() => {});
      },
    });
    await expect(
      nimbleAgentRunResultTool({
        ...cfg,
        client,
        wait: { timeoutMs: 150, pollIntervalMs: 10 },
      }).execute!({ runId: RUN_ID }, CTX),
    ).resolves.toMatchObject({ ready: false, status: 'running', isActive: true });
    expect(calls).toBe(2);
  });

  it('maps a synchronous late poll rejection to the last status snapshot', async () => {
    let calls = 0;
    const client = mockClient({
      get: async () => {
        calls += 1;
        if (calls === 1) return makeRun({ status: 'running', is_active: true });
        const deadline = performance.now() + 300;
        while (performance.now() < deadline) { /* deliberately block */ }
        throw new Error('late poll failure');
      },
    });
    await expect(
      nimbleAgentRunResultTool({
        ...cfg,
        client,
        wait: { timeoutMs: 250, pollIntervalMs: 10 },
      }).execute!({ runId: RUN_ID }, CTX),
    ).resolves.toMatchObject({ ready: false, status: 'running', isActive: true });
    expect(calls).toBe(2);
  });

  it.each(['failed', 'cancelled'] as const)(
    'preserves a late terminal poll snapshot as %s/not-ready',
    async (status) => {
      let calls = 0;
      const client = mockClient({
        get: async () => {
          calls += 1;
          if (calls === 1) return makeRun({ status: 'running', is_active: true });
          const deadline = performance.now() + 300;
          while (performance.now() < deadline) { /* deliberately block */ }
          return makeRun({ status, is_active: false });
        },
      });
      await expect(
        nimbleAgentRunResultTool({
          ...cfg,
          client,
          wait: { timeoutMs: 250, pollIntervalMs: 10 },
        }).execute!({ runId: RUN_ID }, CTX),
      ).resolves.toMatchObject({ ready: false, status, isActive: false });
      expect(calls).toBe(2);
    },
  );

  it('bounds the final result request by the original wait timeout', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async (_runId, _query, options) =>
        await new Promise((_, reject) => {
          options?.signal?.addEventListener('abort', () => reject(options.signal?.reason), {
            once: true,
          });
        }),
    });
    const started = performance.now();
    await expect(
      nimbleAgentRunResultTool({
        ...cfg,
        client,
        wait: { timeoutMs: 150, pollIntervalMs: 100 },
      }).execute!({ runId: RUN_ID }, CTX),
    ).resolves.toMatchObject({
      ready: false,
      runId: RUN_ID,
      agentId: AGENT_ID,
      status: 'completed',
      isActive: false,
    });
    expect(performance.now() - started).toBeLessThan(500);
  });

  it.each([
    ['409', () => httpError(409)],
    ['422', () => httpError(422, makeFailedResult('late failure'))],
    ['generic', () => new Error('late result failure')],
  ] as const)('maps a synchronous late result %s rejection to completed/not-ready', async (_name, error) => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => {
        const deadline = performance.now() + 30;
        while (performance.now() < deadline) { /* deliberately block */ }
        throw error();
      },
    });
    await expect(
      nimbleAgentRunResultTool({
        ...cfg,
        client,
        wait: { timeoutMs: 10, pollIntervalMs: 5 },
      }).execute!({ runId: RUN_ID }, CTX),
    ).resolves.toMatchObject({ ready: false, status: 'completed', isActive: false });
  });

  it('enforces the final deadline when an injected client resolves after abort', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => {
        await new Promise((resolve) => setTimeout(resolve, 40));
        return makeTextResult();
      },
    });
    await expect(
      nimbleAgentRunResultTool({
        ...cfg,
        client,
        wait: { timeoutMs: 20, pollIntervalMs: 10 },
      }).execute!({ runId: RUN_ID }, CTX),
    ).resolves.toMatchObject({
      ready: false,
      status: 'completed',
      isActive: false,
    });
  });

  it('enforces the final deadline when an injected client never settles', async () => {
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => await new Promise<never>(() => {}),
    });
    await expect(
      nimbleAgentRunResultTool({
        ...cfg,
        client,
        wait: { timeoutMs: 20, pollIntervalMs: 10 },
      }).execute!({ runId: RUN_ID }, CTX),
    ).resolves.toMatchObject({ ready: false, status: 'completed', isActive: false });
  });

  it('preserves caller cancellation when an injected client resolves after abort', async () => {
    const controller = new AbortController();
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => {
        await new Promise((resolve) => setTimeout(resolve, 40));
        return makeTextResult();
      },
    });
    const pending = nimbleAgentRunResultTool({
      ...cfg,
      client,
      wait: { timeoutMs: 5_000, pollIntervalMs: 100 },
    }).execute!({ runId: RUN_ID }, { abortSignal: controller.signal } as never);
    setTimeout(() => controller.abort(new Error('caller cancelled')), 10);
    await expect(pending).rejects.toBeInstanceOf(NimbleAgentRunError);
  });

  it('preserves caller cancellation when an injected client never settles', async () => {
    const controller = new AbortController();
    const client = mockClient({
      get: async () => makeRun({ status: 'completed', is_active: false }),
      result: async () => await new Promise<never>(() => {}),
    });
    const pending = nimbleAgentRunResultTool({ ...cfg, client }).execute!(
      { runId: RUN_ID },
      { abortSignal: controller.signal } as never,
    );
    setTimeout(() => controller.abort(new Error('caller cancelled')), 10);
    await expect(pending).rejects.toBeInstanceOf(NimbleAgentRunError);
  });

  it.each(['initial status', 'final result'] as const)(
    'preserves caller cancellation as an error during %s',
    async (phase) => {
      const controller = new AbortController();
      const hanging = async (_runId: string, _query: unknown, options?: { signal?: AbortSignal }) =>
        await new Promise<never>((_, reject) => {
          options?.signal?.addEventListener('abort', () => reject(options.signal?.reason), { once: true });
        });
      const client = mockClient({
        get: phase === 'initial status'
          ? hanging as never
          : async () => makeRun({ status: 'completed', is_active: false }),
        result: phase === 'final result' ? hanging as never : undefined,
      });
      const pending = nimbleAgentRunResultTool({
        ...cfg,
        client,
        wait: { timeoutMs: 5_000, pollIntervalMs: 100 },
      }).execute!({ runId: RUN_ID }, { abortSignal: controller.signal } as never);
      setTimeout(() => controller.abort(new Error('caller cancelled')), 10);
      await expect(pending).rejects.toBeInstanceOf(NimbleAgentRunError);
    },
  );
});
