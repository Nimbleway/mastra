import { describe, expect, it, vi } from 'vitest';
import { inspect } from 'node:util';
import {
  nimbleAgentRunResultTool,
  nimbleAgentRunStatusTool,
  nimbleAgentStartRunTool,
} from '../src/tools';
import { NimbleAgentRunError, NimbleConfigError } from '../src/errors';
import type { NimbleAgentRunCompletedOutput, NimbleAgentRunCreateBody } from '../src/schemas';
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
      nimbleAgentStartRunTool({ agentId: 'wsa_configured', client }).execute!(
        { task: 't' },
        CTX,
      ),
    ).rejects.toMatchObject({ reason: 'protocol' });
  });

  it('fails closed when create returns a different agent id than requested', async () => {
    const client = mockClient({ create: async () => makeRun() });
    await expect(
      nimbleAgentStartRunTool({ agentId: 'wsa_configured', client }).execute!(
        { task: 't' },
        CTX,
      ),
    ).rejects.toMatchObject({ reason: 'protocol', agentId: 'wsa_configured' });
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
});

describe('result tool', () => {
  const cfg = { agentId: AGENT_ID, apiKey: TEST_API_KEY };

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
    await expect(
      nimbleAgentRunResultTool({
        ...cfg,
        client,
        wait: { timeoutMs: 150, pollIntervalMs: 100 },
      }).execute!({ runId: RUN_ID }, CTX),
    ).rejects.toBeInstanceOf(NimbleAgentRunError);
    expect(performance.now() - started).toBeLessThan(500);
  });
});
