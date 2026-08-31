import { createTool } from '@mastra/core/tools';
import { createNimbleClient } from './client';
import {
  NIMBLE_AGENT_RUN_STATUSES,
  NIMBLE_AGENT_EFFORTS,
  nimbleAgentRunIdInputSchema,
  nimbleAgentRunResultOutputSchema,
  nimbleAgentRunStatusOutputSchema,
  nimbleAgentStartRunInputSchema,
  nimbleAgentStartRunOutputSchema,
  nimbleAgentTrustSchema,
} from './schemas';
import type {
  NimbleAgentEffort,
  NimbleAgentOutput,
  NimbleAgentRawFailedResult,
  NimbleAgentRawResult,
  NimbleAgentRawRun,
  NimbleAgentRequestOptions,
  NimbleAgentRunCreateBody,
  NimbleAgentRunCompletedOutput,
  NimbleAgentRunLifecycleStatus,
  NimbleAgentRunPendingOutput,
  NimbleAgentRunResultConfig,
  NimbleAgentRunResultOutput,
  NimbleAgentRunsClient,
  NimbleAgentRunStatusOutput,
  NimbleAgentStartRunConfig,
  NimbleAgentStartRunOutput,
  NimbleAgentToolConfig,
  NimbleAgentToolsConfig,
  NimbleAgentWaitOptions,
} from './schemas';
import { NimbleAgentRunError, NimbleConfigError } from './errors';
import type { NimbleAgentCreateOutcome, NimbleAgentRunErrorReason } from './errors';

/**
 * Agent tool defaults. `effortCap` bounds only the *model's* effort choice;
 * the wait values apply when {@link NimbleAgentRunResultConfig.wait} is
 * enabled (it is off by default — the result tool never blocks unless asked).
 */
export const NIMBLE_AGENT_DEFAULTS = {
  effortCap: 'high',
  waitTimeoutMs: 300_000,
  pollIntervalMs: 10_000,
  minPollIntervalMs: 100,
} as const;

const EFFORT_ORDER: Record<NimbleAgentEffort, number> = {
  low: 0,
  medium: 1,
  high: 2,
  'x-high': 3,
  max: 4,
};

// Derived from the canonical const array so a drift between the type and the
// runtime guard cannot compile.
const LIFECYCLE_STATUSES: ReadonlySet<NimbleAgentRunLifecycleStatus> = new Set(
  NIMBLE_AGENT_RUN_STATUSES,
);
const EFFORTS: ReadonlySet<NimbleAgentEffort> = new Set(NIMBLE_AGENT_EFFORTS);

function capEffort(requested: NimbleAgentEffort, cap: NimbleAgentEffort): NimbleAgentEffort {
  return EFFORT_ORDER[requested] > EFFORT_ORDER[cap] ? cap : requested;
}

function readStatus(err: unknown): number | undefined {
  if (typeof err === 'object' && err !== null && 'status' in err) {
    const status = (err as { status?: unknown }).status;
    return typeof status === 'number' ? status : undefined;
  }
  return undefined;
}

function asFailedResult(value: unknown): NimbleAgentRawFailedResult | undefined {
  if (typeof value !== 'object' || value === null) return undefined;
  const candidate = value as Partial<NimbleAgentRawFailedResult>;
  if (
    typeof candidate.run?.status === 'string' &&
    typeof candidate.error?.message === 'string'
  ) {
    return candidate as NimbleAgentRawFailedResult;
  }
  return undefined;
}

/**
 * Parsed error body of an SDK APIError, when it carries a failed result. The
 * API delivers the 422 failure form either bare (`{ run, error }`) or wrapped
 * in the gateway's error envelope (`{ detail: { run, error } }`) — accept both
 * so terminal failed/cancelled mapping and the server message survive either
 * shape.
 */
function readFailedResultBody(err: unknown): NimbleAgentRawFailedResult | undefined {
  if (typeof err !== 'object' || err === null || !('error' in err)) return undefined;
  const body = (err as { error?: unknown }).error;
  if (typeof body !== 'object' || body === null) return undefined;
  return asFailedResult(body) ?? asFailedResult((body as { detail?: unknown }).detail);
}

interface AgentContext {
  client: NimbleAgentRunsClient;
  agentId: string;
  /** Present only when this package constructed the client (for scrubbing). */
  apiKey?: string;
  /** False for an injected client whose credential is unavailable to scrub. */
  allowErrorDetails: boolean;
}

/**
 * Error messages are built from server response bodies; an auth error is the
 * most likely place for a credential to be echoed back. Scrub the resolved
 * key from any message this package throws — the key must never appear in
 * model-visible output.
 */
function scrub(text: string, apiKey: string | undefined): string {
  return apiKey && text.includes(apiKey) ? text.split(apiKey).join('[redacted]') : text;
}

/** True when the key appears anywhere in the error's message/stack/body chain. */
function keyInErrorChain(err: unknown, apiKey: string, depth = 0): boolean {
  if (err === null || err === undefined) return false;
  if (typeof err !== 'object') return String(err).includes(apiKey);
  // The remainder of an over-deep object chain cannot be proven clean. Fail
  // closed instead of retaining a potentially credential-bearing raw cause.
  if (depth > 4) return true;
  const candidate = err as Error & { cause?: unknown };
  if (typeof candidate.message === 'string' && candidate.message.includes(apiKey)) return true;
  if (typeof candidate.stack === 'string' && candidate.stack.includes(apiKey)) return true;
  try {
    // Enumerable fields (e.g. an APIError's parsed response body) also count.
    if (JSON.stringify(candidate).includes(apiKey)) return true;
  } catch {
    return true; // circular / unserializable — assume dirty rather than leak
  }
  return keyInErrorChain(candidate.cause, apiKey, depth + 1);
}

/**
 * `Error.cause` is printed by `console.error`/`util.inspect` in every
 * downstream consumer, so a raw cause whose message, stack, or response body
 * echoes the key would leak it even though the outer message is scrubbed.
 * Clean causes pass through untouched (full debug fidelity); dirty ones are
 * replaced by a flat, scrubbed copy.
 */
function sanitizeCause(err: unknown, apiKey: string | undefined): unknown {
  if (!apiKey || !keyInErrorChain(err, apiKey)) return err;
  const original = err instanceof Error ? err : undefined;
  const copy = new Error(scrub(original?.message ?? String(err), apiKey));
  copy.name = original?.name ?? 'Error';
  if (original?.stack) copy.stack = scrub(original.stack, apiKey);
  return copy;
}

/** Keep untrusted string metadata only when it is safe to expose. */
function safeErrorMetadata(value: unknown, apiKey: string | undefined): string | undefined {
  if (typeof value !== 'string') return undefined;
  return apiKey && value.includes(apiKey) ? undefined : value;
}

function safeErrorReason(value: unknown): NimbleAgentRunErrorReason {
  return value === 'failed' || value === 'cancelled' || value === 'protocol' || value === 'request'
    ? value
    : 'request';
}

function safeCreateOutcome(value: unknown): NimbleAgentCreateOutcome | undefined {
  return value === 'unknown' || value === 'not-created' ? value : undefined;
}

function toAgentError(
  err: unknown,
  context: {
    verb: string;
    runId?: string;
    agentId?: string;
    apiKey?: string;
    allowErrorDetails: boolean;
  },
): NimbleAgentRunError {
  if (err instanceof NimbleAgentRunError) {
    const runRef = context.runId ? ` (run ${context.runId})` : '';
    const message = context.allowErrorDetails
      ? scrub(err.message, context.apiKey)
      : `Nimble agent ${context.verb} failed${runRef}: details withheld because the injected client credential was not provided for redaction`;
    return new NimbleAgentRunError(message, {
      reason: safeErrorReason(err.reason),
      // Requested identifiers are trusted and authoritative on read paths.
      // Never copy model-visible metadata from an injected error when its
      // credential is unavailable for redaction.
      runId: context.runId,
      agentId: context.agentId,
      runStatus: context.allowErrorDetails
        ? safeErrorMetadata(err.runStatus, context.apiKey)
        : undefined,
      status: typeof err.status === 'number' ? err.status : undefined,
      createOutcome: safeCreateOutcome(err.createOutcome),
      cause: context.allowErrorDetails ? sanitizeCause(err.cause, context.apiKey) : undefined,
    });
  }
  const message = context.allowErrorDetails
    ? scrub(err instanceof Error ? err.message : String(err), context.apiKey)
    : 'details withheld because the injected client credential was not provided for redaction';
  const runRef = context.runId ? ` (run ${context.runId})` : '';
  return new NimbleAgentRunError(`Nimble agent ${context.verb} failed${runRef}: ${message}`, {
    reason: 'request',
    runId: context.runId,
    agentId: context.agentId,
    status: readStatus(err),
    cause: context.allowErrorDetails ? sanitizeCause(err, context.apiKey) : undefined,
  });
}

/**
 * Classify a failed one-shot create. A definite 4xx (429 rate limit, 400/401/
 * 403/404/422 validation or auth) means no run exists — safe to create again
 * once the underlying condition clears. Anything else (timeout, 408, 5xx,
 * connection drop) is ambiguous: the POST may have reached the server and
 * created a billed run whose ID we never received.
 */
function classifyCreateOutcome(err: unknown): NimbleAgentCreateOutcome {
  const status = readStatus(err);
  // 408 is a timeout (request may have landed) and 409 is a state conflict
  // (the server processed something) — both stay 'unknown' so the caller
  // reconciles instead of re-creating. Other 4xx are definite rejections.
  return typeof status === 'number' &&
    status >= 400 &&
    status < 500 &&
    status !== 408 &&
    status !== 409
    ? 'not-created'
    : 'unknown';
}

function toCreateError(
  err: unknown,
  context: { agentId: string; apiKey?: string; allowErrorDetails: boolean },
): NimbleAgentRunError {
  const outcome =
    err instanceof NimbleAgentRunError && safeCreateOutcome(err.createOutcome)
      ? safeCreateOutcome(err.createOutcome)!
      : classifyCreateOutcome(err);
  const message = context.allowErrorDetails
    ? scrub(err instanceof Error ? err.message : String(err), context.apiKey)
    : 'request failed; details withheld because the injected client credential was not provided for redaction';
  const guidance =
    outcome === 'not-created'
      ? 'No run was created; a retry only makes sense after the underlying condition is resolved.'
      : 'The run may or may not have been created server-side. Do not automatically start ' +
        'another run — list recent runs for this agent (or check the Nimble console) to ' +
        'reconcile before creating again.';
  return new NimbleAgentRunError(`Nimble agent run creation failed: ${message} ${guidance}`, {
    reason: err instanceof NimbleAgentRunError ? safeErrorReason(err.reason) : 'request',
    runId:
      context.allowErrorDetails && err instanceof NimbleAgentRunError
        ? safeErrorMetadata(err.runId, context.apiKey)
        : undefined,
    agentId: context.agentId,
    runStatus:
      context.allowErrorDetails && err instanceof NimbleAgentRunError
        ? safeErrorMetadata(err.runStatus, context.apiKey)
        : undefined,
    status:
      err instanceof NimbleAgentRunError && typeof err.status === 'number'
        ? err.status
        : readStatus(err),
    createOutcome: outcome,
    cause: context.allowErrorDetails
      ? sanitizeCause(err instanceof NimbleAgentRunError ? err.cause : err, context.apiKey)
      : undefined,
  });
}

function terminalFailure(
  run: Pick<NimbleAgentRawRun, 'status'> & Partial<NimbleAgentRawRun>,
  ids: { runId: string; agentId: string },
  serverMessage?: string,
  apiKey?: string,
  allowErrorDetails = true,
): NimbleAgentRunError {
  const reason = run.status === 'cancelled' ? 'cancelled' : 'failed';
  // Server-reported failure text is scrubbed like every other error surface —
  // a backend message that echoed credentials must not reach model output.
  const rawDetail = allowErrorDetails ? (serverMessage ?? run.error?.message) : undefined;
  const detail = rawDetail === undefined ? undefined : scrub(rawDetail, apiKey);
  return new NimbleAgentRunError(
    `Nimble agent run ${ids.runId} ${run.status}${detail ? `: ${detail}` : '.'}`,
    {
      reason,
      runId: ids.runId,
      agentId: ids.agentId,
      runStatus: run.status,
    },
  );
}

function resolveAgentContext(config: NimbleAgentToolConfig, factory: string): AgentContext {
  const agentId = config.agentId ?? process.env.NIMBLE_AGENT_ID;
  if (!agentId) {
    throw new NimbleConfigError(
      `Missing Nimble agent id: set NIMBLE_AGENT_ID or pass { agentId } to ${factory}(). ` +
        'Create an agent instance once via the Nimble console or POST /v2/agents.',
    );
  }
  if (config.client) {
    // Only an explicitly paired key can be assumed to belong to an injected
    // client. An ambient NIMBLE_API_KEY may describe a different client and
    // therefore cannot make unknown error details safe to expose.
    const scrubKey = config.apiKey;
    return {
      client: config.client,
      agentId,
      ...(scrubKey ? { apiKey: scrubKey } : {}),
      allowErrorDetails: Boolean(scrubKey),
    };
  }
  const apiKey = config.apiKey ?? process.env.NIMBLE_API_KEY;
  if (!apiKey) {
    throw new NimbleConfigError(
      `Missing Nimble API key: set NIMBLE_API_KEY or pass { apiKey } to ${factory}().`,
    );
  }
  const client = createNimbleClient(
    apiKey,
    config.clientOptions,
  ) as unknown as NimbleAgentRunsClient;
  return { client, agentId, apiKey, allowErrorDetails: true };
}

function requestOptions(signal: AbortSignal | undefined): NimbleAgentRequestOptions | undefined {
  return signal ? { signal } : undefined;
}

function assertKnownStatus(
  run: NimbleAgentRawRun,
  ids: { runId: string; agentId: string },
): void {
  if (!LIFECYCLE_STATUSES.has(run.status)) {
    throw protocolError(ids);
  }
}

function isOptionalString(value: unknown): boolean {
  return value === undefined || value === null || typeof value === 'string';
}

function hasValidRunFields(run: Partial<NimbleAgentRawRun>): boolean {
  const error = run.error;
  return (
    typeof run.id === 'string' &&
    run.id.length > 0 &&
    typeof run.web_search_agent_id === 'string' &&
    run.web_search_agent_id.length > 0 &&
    typeof run.interaction_id === 'string' &&
    run.interaction_id.length > 0 &&
    typeof run.is_active === 'boolean' &&
    EFFORTS.has(run.effort as NimbleAgentEffort) &&
    typeof run.created_at === 'string' &&
    run.created_at.length > 0 &&
    isOptionalString(run.started_at) &&
    isOptionalString(run.completed_at) &&
    isOptionalString(run.prompt) &&
    (error === undefined ||
      error === null ||
      (typeof error === 'object' &&
        typeof error.message === 'string' &&
        typeof error.ref_id === 'string'))
  );
}

/** Fail closed when a read/result body does not belong to the requested run. */
function assertMatchingRunIds(
  run: unknown,
  ids: { runId: string; agentId: string },
): asserts run is NimbleAgentRawRun {
  if (typeof run !== 'object' || run === null) {
    throw protocolError(ids);
  }
  const candidate = run as Partial<NimbleAgentRawRun>;
  if (candidate.id !== ids.runId || candidate.web_search_agent_id !== ids.agentId) {
    throw new NimbleAgentRunError(
      `Nimble agent run ${ids.runId} returned mismatched identifiers.`,
      { reason: 'protocol', runId: ids.runId, agentId: ids.agentId },
    );
  }
  if (!hasValidRunFields(candidate)) {
    throw protocolError(ids);
  }
}

function createProtocolError(agentId: string, runId?: string): NimbleAgentRunError {
  return new NimbleAgentRunError(
    'Nimble agent run creation returned a malformed payload after the request was accepted. ' +
      'The run may have been created server-side; reconcile recent runs before creating again.',
    {
      reason: 'protocol',
      ...(runId ? { runId } : {}),
      agentId,
      createOutcome: 'unknown',
    },
  );
}

function assertCreatedRun(
  run: NimbleAgentRawRun,
  agentId: string,
  apiKey: string | undefined,
): void {
  if (
    !hasValidRunFields(run) ||
    run.web_search_agent_id !== agentId ||
    !LIFECYCLE_STATUSES.has(run.status) ||
    !/^task_run_[A-Za-z0-9_-]+$/.test(run.id) ||
    (apiKey !== undefined &&
      [
        run.id,
        run.interaction_id,
        run.created_at,
        run.started_at,
        run.completed_at,
        run.prompt,
        run.error?.message,
        run.error?.ref_id,
      ].some((value) => typeof value === 'string' && value.includes(apiKey)))
  ) {
    throw createProtocolError(agentId);
  }
}

function baseFields(run: NimbleAgentRawRun) {
  return {
    runId: run.id,
    // Server-returned IDs are authoritative. Read/result paths reconcile them
    // against the requested IDs before this mapper is called.
    agentId: run.web_search_agent_id,
    effort: run.effort,
    createdAt: run.created_at,
  };
}

function toStartOutput(run: NimbleAgentRawRun): NimbleAgentStartRunOutput {
  return {
    ...baseFields(run),
    interactionId: run.interaction_id,
    status: run.status,
  };
}

function toStatusOutput(
  run: NimbleAgentRawRun,
  apiKey?: string,
  allowErrorDetails = true,
): NimbleAgentRunStatusOutput {
  return {
    ...baseFields(run),
    status: run.status,
    isActive: run.is_active,
    ...(run.started_at ? { startedAt: run.started_at } : {}),
    ...(run.completed_at ? { completedAt: run.completed_at } : {}),
    // Server-reported error text is model-visible output — scrub it.
    ...(allowErrorDetails && run.error?.message
      ? { error: { message: scrub(run.error.message, apiKey) } }
      : {}),
  };
}

function toPendingOutput(
  run: NimbleAgentRawRun,
  status: 'queued' | 'running',
): NimbleAgentRunPendingOutput {
  return {
    ready: false,
    ...baseFields(run),
    status,
    isActive: true,
    ...(run.started_at ? { startedAt: run.started_at } : {}),
  };
}

function protocolError(ids: { runId: string; agentId: string }, runStatus?: string) {
  return new NimbleAgentRunError(
    `Nimble agent run ${ids.runId} returned a malformed result payload.`,
    { reason: 'protocol', runId: ids.runId, agentId: ids.agentId, runStatus },
  );
}

function toAgentOutput(
  raw: NimbleAgentRawResult['output'],
  ids: { runId: string; agentId: string },
): NimbleAgentOutput {
  // The SDK does not runtime-validate bodies; a misbehaving proxy can deliver
  // an out-of-contract container. Keep such cases inside the typed
  // protocol-error contract instead of surfacing a raw TypeError.
  if (typeof raw !== 'object' || raw === null) {
    throw protocolError(ids, 'completed');
  }
  // `trust` is required on both output forms; a body missing it would make the
  // typed output lie (non-null field holding undefined).
  const parsedTrust = nimbleAgentTrustSchema.safeParse(raw.trust);
  if (!parsedTrust.success) {
    throw protocolError(ids, 'completed');
  }
  const kind = raw.type ?? (typeof raw.content === 'string' ? 'text' : 'json');
  if (kind === 'text' && typeof raw.content === 'string') {
    return { type: 'text', text: raw.content, trust: parsedTrust.data };
  }
  if (kind === 'json' && typeof raw.content === 'object' && raw.content !== null) {
    return { type: 'json', json: raw.content, trust: parsedTrust.data };
  }
  throw protocolError(ids, 'completed');
}

function toCompletedOutput(
  result: NimbleAgentRawResult,
): NimbleAgentRunCompletedOutput {
  const run = result.run;
  return {
    ready: true,
    ...baseFields(run),
    status: 'completed',
    ...(run.started_at ? { startedAt: run.started_at } : {}),
    ...(run.completed_at ? { completedAt: run.completed_at } : {}),
    output: toAgentOutput(result.output, {
      runId: run.id,
      agentId: run.web_search_agent_id,
    }),
  };
}

/** Abortable sleep; rejects with the signal's reason when aborted. */
function sleep(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortReason(signal));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    function onAbort() {
      clearTimeout(timer);
      reject(abortReason(signal));
    }
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

function abortReason(signal: AbortSignal | undefined): unknown {
  return signal?.reason ?? new Error('The operation was aborted.');
}

/**
 * A finite, positive number, or the fallback. Guards against non-finite /
 * non-positive wait values (notably `NaN` from `Number(unset env)`) — which
 * `??` would treat as "provided", leaving a `NaN` timeout that never trips the
 * `remaining <= 0` break and a `NaN` sleep that coerces to a 0ms tight poll.
 */
function positiveFinite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback;
}

function normalizeWait(
  wait: NimbleAgentRunResultConfig['wait'],
): Required<NimbleAgentWaitOptions> | undefined {
  if (!wait) return undefined;
  const options = wait === true ? {} : wait;
  return {
    timeoutMs: positiveFinite(options.timeoutMs, NIMBLE_AGENT_DEFAULTS.waitTimeoutMs),
    pollIntervalMs: Math.max(
      positiveFinite(options.pollIntervalMs, NIMBLE_AGENT_DEFAULTS.pollIntervalMs),
      NIMBLE_AGENT_DEFAULTS.minPollIntervalMs,
    ),
  };
}

/**
 * Create a Mastra tool that starts a Nimble deep-research agent run
 * (`@nimble-way/nimble-js` → `POST /v2/agents/{agent_id}/runs`) and returns
 * immediately with the real `task_run_…` ID.
 *
 * The research runs asynchronously on Nimble's side (typically minutes at
 * `medium`+ effort) — an agent turn is never blocked for the run's duration.
 * Pair with {@link nimbleAgentRunStatusTool} / {@link nimbleAgentRunResultTool}
 * to collect the answer later, from the same or a completely different process.
 *
 * The POST is sent exactly once (`maxRetries: 0`): run creation is billed and
 * not idempotent, so transient failures surface as errors with reconciliation
 * guidance instead of being silently replayed.
 */
export function nimbleAgentStartRunTool(config: NimbleAgentStartRunConfig = {}) {
  const effortCap = config.effortCap ?? NIMBLE_AGENT_DEFAULTS.effortCap;

  return createTool({
    id: 'nimble-agent-start-run',
    description:
      'Start a Nimble deep-research agent run for a complex research task. ' +
      'Returns immediately with a runId while the research (which can take ' +
      'minutes) continues in the background. Use the nimble-agent-run-status ' +
      'or nimble-agent-run-result tool with the returned runId to collect the ' +
      'answer later — even from a different conversation turn or process.',
    inputSchema: nimbleAgentStartRunInputSchema,
    outputSchema: nimbleAgentStartRunOutputSchema,
    execute: async (input, context): Promise<NimbleAgentStartRunOutput> => {
      const { client, agentId, apiKey, allowErrorDetails } = resolveAgentContext(
        config,
        'nimbleAgentStartRunTool',
      );
      const signal = context?.abortSignal;

      const effort = input.effort ? capEffort(input.effort, effortCap) : config.effort;
      const body: NimbleAgentRunCreateBody = {
        input: input.task,
        ...(effort ? { effort } : {}),
      };

      let run: NimbleAgentRawRun;
      try {
        // One-shot by contract: never let the SDK's default retry policy
        // (408/409/429/5xx) replay a non-idempotent, billed POST.
        run = await client.agents.runs.create(agentId, body, {
          ...requestOptions(signal),
          maxRetries: 0,
        });
      } catch (err) {
        throw toCreateError(err, { agentId, apiKey, allowErrorDetails });
      }
      if (typeof run !== 'object' || run === null) {
        throw createProtocolError(agentId);
      }
      assertCreatedRun(run, agentId, apiKey);
      return toStartOutput(run);
    },
  });
}

/**
 * Create a Mastra tool that reports the current status of a Nimble agent run
 * (`GET /v2/agents/{agent_id}/runs/{run_id}`). Instant and cheap; never
 * waits. Works for any run of the configured agent, including runs started by
 * another process — only the `runId` is needed.
 */
export function nimbleAgentRunStatusTool(config: NimbleAgentToolConfig = {}) {
  return createTool({
    id: 'nimble-agent-run-status',
    description:
      'Check the current status of a Nimble deep-research agent run by runId ' +
      '(queued, running, completed, failed, or cancelled). Instant; never ' +
      'waits. Use the nimble-agent-run-result tool to fetch the finished answer.',
    inputSchema: nimbleAgentRunIdInputSchema,
    outputSchema: nimbleAgentRunStatusOutputSchema,
    execute: async (input, context): Promise<NimbleAgentRunStatusOutput> => {
      const { client, agentId, apiKey, allowErrorDetails } = resolveAgentContext(
        config,
        'nimbleAgentRunStatusTool',
      );
      const signal = context?.abortSignal;

      let run: NimbleAgentRawRun;
      try {
        run = await client.agents.runs.get(
          input.runId,
          { agent_id: agentId },
          requestOptions(signal),
        );
      } catch (err) {
        throw toAgentError(err, {
          verb: 'status check',
          runId: input.runId,
          agentId,
          apiKey,
          allowErrorDetails,
        });
      }
      assertMatchingRunIds(run, { runId: input.runId, agentId });
      assertKnownStatus(run, { runId: input.runId, agentId });
      return toStatusOutput(run, apiKey, allowErrorDetails);
    },
  });
}

/**
 * Create a Mastra tool that fetches the result of a Nimble agent run
 * (`GET /v2/agents/{agent_id}/runs/{run_id}/result`).
 *
 * A still-active run returns `{ ready: false, status }` — an expected async
 * state, not an error — so the agent can tell the user to check back. Enable
 * `config.wait` to bounded-poll first (timeout + AbortSignal aware; the run
 * keeps going server-side if the wait gives up). A run that terminally
 * `failed`/`cancelled` throws {@link NimbleAgentRunError} with the runId
 * preserved. Completed runs return the answer — prose `text` or structured
 * `json` — plus verbatim `trust` metadata (sources, per-claim citations,
 * excerpts, confidence).
 */
export function nimbleAgentRunResultTool(config: NimbleAgentRunResultConfig = {}) {
  const wait = normalizeWait(config.wait);

  return createTool({
    id: 'nimble-agent-run-result',
    description:
      'Fetch the result of a Nimble deep-research agent run by runId. If the ' +
      'run is still working, returns { ready: false } — check again later. ' +
      'When complete, returns the answer (text or structured JSON) with ' +
      'sources, per-claim citations, and confidence metadata.',
    inputSchema: nimbleAgentRunIdInputSchema,
    outputSchema: nimbleAgentRunResultOutputSchema,
    execute: async (input, context): Promise<NimbleAgentRunResultOutput> => {
      const { client, agentId, apiKey, allowErrorDetails } = resolveAgentContext(
        config,
        'nimbleAgentRunResultTool',
      );
      const signal = context?.abortSignal;
      const ids = { runId: input.runId, agentId };

      const getRun = async (): Promise<NimbleAgentRawRun> => {
        try {
          const fetched = await client.agents.runs.get(
            input.runId,
            { agent_id: agentId },
            requestOptions(signal),
          );
          assertMatchingRunIds(fetched, ids);
          return fetched;
        } catch (err) {
          throw toAgentError(err, {
            verb: 'status check',
            ...ids,
            apiKey,
            allowErrorDetails,
          });
        }
      };

      let run = await getRun();
      assertKnownStatus(run, ids);

      if (wait && run.is_active) {
        const startedWaiting = performance.now();
        while (run.is_active) {
          const elapsed = performance.now() - startedWaiting;
          const remaining = wait.timeoutMs - elapsed;
          if (remaining <= 0) break;
          await sleep(Math.min(wait.pollIntervalMs, remaining), signal);
          run = await getRun();
          assertKnownStatus(run, ids);
        }
      }

      if (run.status === 'queued' || run.status === 'running') {
        return toPendingOutput(run, run.status);
      }
      if (run.status === 'failed' || run.status === 'cancelled') {
        throw terminalFailure(run, ids, undefined, apiKey, allowErrorDetails);
      }

      // status === 'completed' — fetch the output.
      let result: NimbleAgentRawResult | NimbleAgentRawFailedResult;
      try {
        result = await client.agents.runs.result(
          input.runId,
          { agent_id: agentId },
          requestOptions(signal),
        );
      } catch (err) {
        const httpStatus = readStatus(err);
        // 409: the result endpoint still considers the run active (eventual
        // consistency with the status we just read) — report not-ready.
        if (httpStatus === 409) {
          return toPendingOutput({ ...run, status: 'running', is_active: true }, 'running');
        }
        // 422: terminal failure — the body carries the run + structured error.
        if (httpStatus === 422) {
          const failed = readFailedResultBody(err);
          if (failed) {
            assertMatchingRunIds(failed.run, ids);
            if (failed.run.status !== 'failed' && failed.run.status !== 'cancelled') {
              throw protocolError(ids);
            }
            throw terminalFailure(
              failed.run,
              ids,
              failed.error.message,
              apiKey,
              allowErrorDetails,
            );
          }
        }
        throw toAgentError(err, {
          verb: 'result fetch',
          ...ids,
          apiKey,
          allowErrorDetails,
        });
      }

      if (typeof result !== 'object' || result === null) {
        throw protocolError(ids);
      }
      if (!('output' in result)) {
        // Failed form: { run, error }. Require the complete envelope and a
        // failed/cancelled terminal status before mapping it as a run failure.
        const failed = asFailedResult(result);
        if (
          !failed ||
          (failed.run.status !== 'failed' && failed.run.status !== 'cancelled')
        ) {
          throw protocolError(ids);
        }
        assertMatchingRunIds(failed.run, ids);
        throw terminalFailure(
          failed.run,
          ids,
          failed.error.message,
          apiKey,
          allowErrorDetails,
        );
      }
      // Re-validate the run object embedded in the result payload rather than
      // trusting only the earlier status snapshot: an eventually-inconsistent
      // or malformed body must not be stamped `completed` by toCompletedOutput.
      const resultRun = result.run;
      if (typeof resultRun?.status !== 'string') throw protocolError(ids);
      assertMatchingRunIds(resultRun, ids);
      assertKnownStatus(resultRun, ids);
      if (resultRun.status === 'queued' || resultRun.status === 'running') {
        return toPendingOutput(resultRun, resultRun.status);
      }
      if (resultRun.status === 'failed' || resultRun.status === 'cancelled') {
        throw terminalFailure(resultRun, ids, undefined, apiKey, allowErrorDetails);
      }
      return toCompletedOutput(result);
    },
  });
}

/**
 * Convenience factory: build all three run-lifecycle tools sharing one
 * server-only API key, agent id, and client configuration. The returned map
 * plugs directly into a Mastra `Agent`'s `tools`:
 *
 * ```ts
 * const agent = new Agent({
 *   // …model, instructions…
 *   tools: createNimbleAgentTools({ agentId: process.env.NIMBLE_AGENT_ID }),
 * });
 * ```
 *
 * When this factory constructs the client (no `client` injected), the three
 * tools share a single underlying Nimble client instance.
 */
export function createNimbleAgentTools(config: NimbleAgentToolsConfig = {}) {
  // Share one lazily-created client across the three tools so credentials are
  // resolved once and connection reuse applies. Lazy so the factory itself
  // never throws in key-less environments.
  let shared: NimbleAgentRunsClient | undefined;
  let sharedApiKey: string | undefined;
  const sharedConfig: NimbleAgentToolsConfig = {
    ...config,
    get apiKey(): string | undefined {
      if (config.client) return config.apiKey;
      return sharedApiKey ?? config.apiKey ?? process.env.NIMBLE_API_KEY;
    },
    get client(): NimbleAgentRunsClient | undefined {
      if (config.client) return config.client;
      if (shared) return shared;
      const apiKey = config.apiKey ?? process.env.NIMBLE_API_KEY;
      if (!apiKey) return undefined; // let resolveAgentContext raise NimbleConfigError
      sharedApiKey = apiKey;
      shared = createNimbleClient(
        apiKey,
        config.clientOptions,
      ) as unknown as NimbleAgentRunsClient;
      return shared;
    },
  };

  return {
    nimbleAgentStartRun: nimbleAgentStartRunTool(sharedConfig),
    nimbleAgentRunStatus: nimbleAgentRunStatusTool(sharedConfig),
    nimbleAgentRunResult: nimbleAgentRunResultTool(sharedConfig),
  };
}
