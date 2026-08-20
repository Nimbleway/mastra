import { describe, expect, it, vi } from 'vitest';
import { nimbleAgentStartRunTool, nimbleAgentRunStatusTool } from '../src/tools';
import { NimbleAgentRunError } from '../src/errors';
import { AGENT_ID, CTX, FAKE_KEY, jsonResponse, makeRun } from './fixtures';

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
