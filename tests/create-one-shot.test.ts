import { describe, expect, it, vi } from 'vitest';
import { inspect } from 'node:util';
import { nimbleAgentStartRunTool, nimbleAgentRunStatusTool } from '../src/tools';
import { NimbleAgentRunError } from '../src/errors';
import { AGENT_ID, CTX, FAKE_KEY, RUN_ID, jsonResponse, makeRun, mockClient } from './fixtures';

/**
 * Attempt-count contract, proven against the REAL `@nimble-way/nimble-js`
 * client (constructed by the package) with a mock `fetch`: run creation must
 * be one-shot (`maxRetries: 0`) for every status the SDK would otherwise
 * retry (408, 409, 429, 5xx), while read requests keep their retries.
 */
describe('run creation is one-shot', () => {
  function startToolWithFetch(fetchImpl: typeof fetch, maxRetries?: number) {
    return nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      clientOptions: { fetch: fetchImpl, ...(maxRetries !== undefined ? { maxRetries } : {}) },
    });
  }

  for (const status of [408, 409, 429, 500, 503]) {
    it(`sends exactly 1 attempt on HTTP ${status}`, async () => {
      const fetchMock = vi.fn(async () => jsonResponse(status, { detail: 'boom' }));
      const tool = startToolWithFetch(fetchMock as unknown as typeof fetch);
      await expect(tool.execute!({ task: 'research x' }, CTX)).rejects.toThrow(
        NimbleAgentRunError,
      );
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });
  }

  it('sends exactly 1 attempt even when clientOptions.maxRetries is raised', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(500, { detail: 'boom' }));
    const tool = startToolWithFetch(fetchMock as unknown as typeof fetch, 3);
    await expect(tool.execute!({ task: 'research x' }, CTX)).rejects.toThrow(NimbleAgentRunError);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('control: status reads DO retry (per-request 0 applies to create only)', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(500, { detail: 'boom' }));
    const tool = nimbleAgentRunStatusTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      clientOptions: { fetch: fetchMock as unknown as typeof fetch, maxRetries: 1 },
    });
    await expect(tool.execute!({ runId: 'task_run_x' }, CTX)).rejects.toThrow(NimbleAgentRunError);
    expect(fetchMock).toHaveBeenCalledTimes(2); // initial + 1 retry
  });

  it('does not retry a connection error either', async () => {
    const fetchMock = vi.fn(async () => {
      throw new TypeError('fetch failed: connection reset');
    });
    const tool = startToolWithFetch(fetchMock as unknown as typeof fetch);
    await expect(tool.execute!({ task: 'research x' }, CTX)).rejects.toThrow(NimbleAgentRunError);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['never settles', async () => await new Promise<never>(() => undefined)],
    ['settles late', async () => await new Promise<ReturnType<typeof makeRun>>((resolve) => setTimeout(() => resolve(makeRun()), 50))],
  ] as const)('bounds a cancelled injected create that %s', async (_name, create) => {
    const controller = new AbortController();
    let attempts = 0;
    const tool = nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      client: mockClient({ create: async () => { attempts += 1; return await create(); } }),
    });
    const pending = tool.execute!({ task: 'research x' }, { abortSignal: controller.signal } as never);
    controller.abort(new Error('caller cancelled'));
    await expect(pending).rejects.toMatchObject({ createOutcome: 'unknown' });
    expect(attempts).toBe(1);
  });

  it('keeps a post-invocation cancellation ambiguous even when its reason resembles a 4xx response', async () => {
    const controller = new AbortController();
    const reason = Object.assign(new Error('caller cancelled'), { status: 429 });
    let attempts = 0;
    const tool = nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      client: mockClient({
        create: async () => {
          attempts += 1;
          return await new Promise<never>(() => undefined);
        },
      }),
    });
    const pending = tool.execute!({ task: 'research x' }, { abortSignal: controller.signal } as never);
    controller.abort(reason);
    await expect(pending).rejects.toMatchObject({ createOutcome: 'unknown' });
    expect(attempts).toBe(1);
  });

  it('rechecks cancellation after sanitizing a hostile create error', async () => {
    const controller = new AbortController();
    let attempts = 0;
    const hostile = new NimbleAgentRunError('rate limited', {
      reason: 'request',
      agentId: AGENT_ID,
      runStatus: 'queued',
      status: 429,
      createOutcome: 'not-created',
    });
    Object.defineProperties(hostile, {
      message: {
        configurable: true,
        get() {
          controller.abort(new Error('caller cancelled'));
          return 'rate limited';
        },
      },
    });
    const tool = nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({
        create: async () => {
          attempts += 1;
          throw hostile;
        },
      }),
    });
    const err = await tool.execute!(
      { task: 'research x' }, { abortSignal: controller.signal } as never,
    ).then(
      () => { throw new Error('expected failure'); },
      (error: unknown) => error as NimbleAgentRunError,
    );
    expect(err).toMatchObject({ createOutcome: 'unknown', runStatus: 'queued', status: 429 });
    expect(err.message).toContain('Do not automatically start another run');
    expect(attempts).toBe(1);
  });

  it('does not invoke create when the caller signal is already aborted', async () => {
    const controller = new AbortController();
    controller.abort(new Error('caller cancelled'));
    let attempts = 0;
    const tool = nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      client: mockClient({ create: async () => { attempts += 1; return makeRun(); } }),
    });
    await expect(
      tool.execute!({ task: 'research x' }, { abortSignal: controller.signal } as never),
    ).rejects.toMatchObject({ createOutcome: 'unknown' });
    expect(attempts).toBe(0);
  });

  it('prioritizes cancellation triggered while snapshotting a create response', async () => {
    const controller = new AbortController();
    let attempts = 0;
    const run = new Proxy(makeRun(), {
      ownKeys(target) {
        controller.abort(new Error('caller cancelled'));
        return Reflect.ownKeys(target);
      },
    });
    const tool = nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      client: mockClient({ create: async () => { attempts += 1; return run; } }),
    });
    await expect(tool.execute!(
      { task: 'research x' }, { abortSignal: controller.signal } as never,
    )).rejects.toMatchObject({ createOutcome: 'unknown', runId: RUN_ID });
    expect(attempts).toBe(1);
  });

  it('sanitizes hostile create snapshot failures as ambiguous', async () => {
    const run = new Proxy(makeRun(), {
      ownKeys() { throw new Error(`snapshot exposed ${FAKE_KEY}`); },
    });
    const err = await nimbleAgentStartRunTool({
      agentId: AGENT_ID, apiKey: FAKE_KEY,
      client: mockClient({ create: async () => run }),
    }).execute!({ task: 'research x' }, CTX).catch((error: unknown) => error as NimbleAgentRunError);
    expect(err).toMatchObject({ reason: 'protocol', createOutcome: 'unknown' });
    for (const rendered of [String(err), JSON.stringify(err), inspect(err, { depth: 10 })]) {
      expect(rendered).not.toContain(FAKE_KEY);
    }
  });

  it('sends X-Client-Source: mastra and the create body on success', async () => {
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      const headers = new Headers(init?.headers);
      expect(headers.get('X-Client-Source')).toBe('mastra');
      expect(String(url)).toContain(`/v2/agents/${AGENT_ID}/runs`);
      expect(init?.method?.toUpperCase()).toBe('POST');
      expect(JSON.parse(String(init?.body))).toEqual({ input: 'research x', effort: 'low' });
      return jsonResponse(200, makeRun());
    });
    const tool = startToolWithFetch(fetchMock as unknown as typeof fetch);
    const out = await tool.execute!({ task: 'research x', effort: 'low' }, CTX);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(out).toMatchObject({ runId: makeRun().id, agentId: AGENT_ID, status: 'queued' });
  });

  it.each([
    null,
    {},
    makeRun({ id: '' }),
    makeRun({ interaction_id: undefined as never }),
    makeRun({ effort: 'unknown' as never }),
    makeRun({ created_at: undefined as never }),
  ])('maps malformed successful create response %# to a protocol error', async (body) => {
    const fetchMock = vi.fn(async () => jsonResponse(200, body));
    const err = await startToolWithFetch(fetchMock as unknown as typeof fetch)
      .execute!({ task: 'research x' }, CTX)
      .then(
        () => { throw new Error('expected failure'); },
        (error: unknown) => error as NimbleAgentRunError,
      );
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(err).toMatchObject({
      reason: 'protocol',
      agentId: AGENT_ID,
      createOutcome: 'unknown',
    });
    expect(err.message).toContain('reconcile recent runs before creating again');
    expect(err).not.toBeInstanceOf(TypeError);
  });

  it('does not expose an untrusted status from an accepted create response', async () => {
    const leakyStatus = `status-${FAKE_KEY}`;
    const fetchMock = vi.fn(async () => jsonResponse(200, makeRun({ status: leakyStatus as never })));
    const err = await startToolWithFetch(fetchMock as unknown as typeof fetch)
      .execute!({ task: 'research x' }, CTX)
      .then(
        () => { throw new Error('expected failure'); },
        (error: unknown) => error as NimbleAgentRunError,
      );

    expect(err).toMatchObject({
      reason: 'protocol',
      agentId: AGENT_ID,
      createOutcome: 'unknown',
    });
    expect(err.runId).toBe(RUN_ID);
    expect(err.runStatus).toBeUndefined();
    expect(err.message).not.toContain(leakyStatus);
    expect(err.message).not.toContain(FAKE_KEY);
  });

  it('does not expose a credential-bearing run id from an accepted create response', async () => {
    const leakyRunId = `task_run_${FAKE_KEY}`;
    const fetchMock = vi.fn(async () =>
      jsonResponse(200, makeRun({ id: leakyRunId, status: 'unknown' as never })),
    );
    const err = await startToolWithFetch(fetchMock as unknown as typeof fetch)
      .execute!({ task: 'research x' }, CTX)
      .then(
        () => { throw new Error('expected failure'); },
        (error: unknown) => error as NimbleAgentRunError,
      );

    expect(err).toMatchObject({
      reason: 'protocol',
      agentId: AGENT_ID,
      createOutcome: 'unknown',
    });
    expect(err.runId).toBeUndefined();
    expect(JSON.stringify(err)).not.toContain(FAKE_KEY);
  });

  it('preserves a safe accepted run id when another create field is malformed', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse(200, makeRun({ effort: 'unexpected' as never })),
    );
    const err = await startToolWithFetch(fetchMock as unknown as typeof fetch)
      .execute!({ task: 'research x' }, CTX)
      .then(
        () => { throw new Error('expected failure'); },
        (error: unknown) => error as NimbleAgentRunError,
      );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(err).toMatchObject({
      reason: 'protocol',
      runId: RUN_ID,
      agentId: AGENT_ID,
      createOutcome: 'unknown',
    });
    expect(err.message).toContain(RUN_ID);
  });

  it('withholds an accepted run id when the response belongs to another agent', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse(200, makeRun({ web_search_agent_id: 'wsa_other' })),
    );
    const err = await startToolWithFetch(fetchMock as unknown as typeof fetch)
      .execute!({ task: 'research x' }, CTX)
      .then(
        () => { throw new Error('expected failure'); },
        (error: unknown) => error as NimbleAgentRunError,
      );

    expect(err).toMatchObject({ reason: 'protocol', agentId: AGENT_ID });
    expect(err.runId).toBeUndefined();
  });

  it.each(['id', 'web_search_agent_id'] as const)(
    'rejects an accepted create response with a throwing %s accessor',
    async (property) => {
      const run = makeRun();
      Object.defineProperty(run, property, {
        get() { throw new Error(`${property} exposed ${FAKE_KEY}`); },
      });
      const err = await nimbleAgentStartRunTool({
        agentId: AGENT_ID,
        apiKey: FAKE_KEY,
        client: mockClient({ create: async () => run }),
      }).execute!({ task: 'research x' }, CTX).then(
        () => { throw new Error('expected failure'); },
        (error: unknown) => error as NimbleAgentRunError,
      );
      expect(err).toMatchObject({
        reason: 'protocol',
        agentId: AGENT_ID,
        createOutcome: 'unknown',
      });
      expect(err.runId).toBeUndefined();
      expect(inspect(err, { depth: 20 })).not.toContain(FAKE_KEY);
    },
  );
});

describe('ambiguous-outcome guidance', () => {
  function failingTool(status: number) {
    const fetchMock = vi.fn(async () => jsonResponse(status, { detail: 'boom' }));
    return nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      clientOptions: { fetch: fetchMock as unknown as typeof fetch },
    });
  }

  async function captureError(status: number): Promise<NimbleAgentRunError> {
    try {
      await failingTool(status).execute!({ task: 't' }, CTX);
    } catch (err) {
      return err as NimbleAgentRunError;
    }
    throw new Error('expected create to fail');
  }

  // 408 (timeout) and 409 (conflict — the server processed *something*) both
  // stay ambiguous: without an idempotency key, reconcile before re-creating.
  for (const status of [408, 409, 500, 503]) {
    it(`HTTP ${status}: outcome unknown — tells the caller to reconcile, not re-create`, async () => {
      const err = await captureError(status);
      expect(err.createOutcome).toBe('unknown');
      expect(err.message).toContain('may or may not have been created');
      expect(err.message).toContain('Do not automatically start another run');
      expect(err.status).toBe(status);
    });
  }

  it.each([408, 409, 500, 503])(
    'HTTP %s overrides contradictory injected not-created metadata as unknown',
    async (status) => {
      const injected = new NimbleAgentRunError('contradictory create failure', {
        reason: 'request',
        status,
        createOutcome: 'not-created',
      });
      const tool = nimbleAgentStartRunTool({
        agentId: AGENT_ID,
        apiKey: FAKE_KEY,
        client: mockClient({ create: async () => { throw injected; } }),
      });
      const err = await tool.execute!({ task: 't' }, CTX)
        .catch((caught: unknown) => caught) as NimbleAgentRunError;
      expect(err).toMatchObject({ status, createOutcome: 'unknown' });
      expect(err.message).toContain('Do not automatically start another run');
    },
  );

  it('reads a stateful create-error status once for classification and metadata', async () => {
    const injected = new NimbleAgentRunError('stateful status', {
      reason: 'request',
      createOutcome: 'not-created',
    });
    let reads = 0;
    Object.defineProperty(injected, 'status', {
      get() { reads += 1; return reads === 1 ? 400 : 503; },
    });
    const tool = nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      client: mockClient({ create: async () => { throw injected; } }),
    });
    const err = await tool.execute!({ task: 't' }, CTX)
      .catch((caught: unknown) => caught) as NimbleAgentRunError;
    expect(reads).toBe(1);
    expect(err).toMatchObject({ status: 400, createOutcome: 'not-created' });
    expect(err.message).toContain('No run was created');
  });

  for (const status of [429, 422, 400, 401]) {
    it(`HTTP ${status}: outcome not-created — definite rejection`, async () => {
      const err = await captureError(status);
      expect(err.createOutcome).toBe('not-created');
      expect(err.message).toContain('No run was created');
    });
  }

  it('connection error (no HTTP status): outcome unknown', async () => {
    const fetchMock = vi.fn(async () => {
      throw new TypeError('fetch failed');
    });
    const tool = nimbleAgentStartRunTool({
      agentId: AGENT_ID,
      apiKey: FAKE_KEY,
      clientOptions: { fetch: fetchMock as unknown as typeof fetch },
    });
    const err = await tool.execute!({ task: 't' }, CTX).then(
      () => {
        throw new Error('expected failure');
      },
      (e: unknown) => e as NimbleAgentRunError,
    );
    expect(err.createOutcome).toBe('unknown');
    expect(err.message).toContain('may or may not have been created');
  });
});
