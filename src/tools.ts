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
  NimbleAgentTrust,
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
  try {
    if (typeof err === 'object' && err !== null && 'status' in err) {
      const status = (err as { status?: unknown }).status;
      return typeof status === 'number' ? status : undefined;
    }
  } catch { /* untrusted runtime accessor */ }
  return undefined;
}

function asFailedResult(value: unknown): NimbleAgentRawFailedResult | undefined {
  try {
    const snapshot = snapshotPlainData(value);
    if (!snapshot.ok || typeof snapshot.value !== 'object' || snapshot.value === null) {
      return undefined;
    }
    const candidate = snapshot.value as Partial<NimbleAgentRawFailedResult>;
    if (
      typeof candidate.run?.status === 'string' &&
      typeof candidate.error?.message === 'string' &&
      typeof candidate.error.ref_id === 'string'
    ) {
      return candidate as NimbleAgentRawFailedResult;
    }
  } catch { /* hostile response accessor */ }
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
  const body = safeErrorProperty(err, 'error');
  if (typeof body !== 'object' || body === null) return undefined;
  return asFailedResult(body) ?? asFailedResult(safeErrorProperty(body, 'detail'));
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

function isErrorObject(value: unknown): value is Error {
  try {
    return value instanceof Error;
  } catch {
    return false;
  }
}

function isNimbleRunError(value: unknown): value is NimbleAgentRunError {
  try {
    return value instanceof NimbleAgentRunError;
  } catch {
    return false;
  }
}

/**
 * `Error.cause` is printed by `console.error`/`util.inspect` in every
 * downstream consumer, so a raw cause whose message, stack, or response body
 * echoes the key would leak it even though the outer message is scrubbed.
 * Causes are always replaced by a flat, scrubbed snapshot. Retaining even a
 * currently clean client-owned Error would let the client mutate it after
 * validation and expose the credential through the wrapper later.
 */
function sanitizeCause(err: unknown, apiKey: string | undefined): unknown {
  if (!apiKey) return undefined;
  const original = isErrorObject(err) ? err : undefined;
  let message = 'Untrusted error details withheld';
  let name = 'Error';
  let stack: string | undefined;
  try {
    const value = original?.message;
    message = scrub(typeof value === 'string' ? value : String(err), apiKey);
  } catch { /* keep safe default */ }
  try {
    const value = original?.name;
    name = scrub(typeof value === 'string' ? value : 'Error', apiKey);
  } catch { /* keep safe default */ }
  try {
    const value = original?.stack;
    if (typeof value === 'string') stack = scrub(value, apiKey);
  } catch { /* omit unsafe stack */ }
  const copy = new Error(message);
  copy.name = name;
  if (stack) copy.stack = stack;
  return copy;
}

/** Keep untrusted string metadata only when it is safe to expose. */
function safeErrorMetadata(value: unknown, apiKey: string | undefined): string | undefined {
  if (typeof value !== 'string') return undefined;
  return apiKey && value.includes(apiKey) ? undefined : value;
}

function assertSafeRequestedRunId(runId: string, agentId: string, apiKey: string | undefined): void {
  if (apiKey && runId.includes(apiKey)) {
    throw new NimbleAgentRunError('Nimble agent run identifier contains protected credential material.', {
      reason: 'protocol',
      agentId,
    });
  }
}

/** Read runtime error properties without trusting user-defined accessors. */
function safeErrorProperty(err: unknown, key: string): unknown {
  if ((typeof err !== 'object' && typeof err !== 'function') || err === null) return undefined;
  try {
    return (err as Record<string, unknown>)[key];
  } catch {
    return undefined;
  }
}

function safeErrorMessage(err: unknown, apiKey: string | undefined): string {
  const message = safeErrorProperty(err, 'message');
  if (typeof message === 'string') return scrub(message, apiKey);
  try {
    return scrub(String(err), apiKey);
  } catch {
    return 'Untrusted error details withheld';
  }
}

/** Inspect raw values so JSON escaping cannot hide a reflected credential. */
function containsCredential(
  value: unknown,
  apiKey: string | undefined,
): boolean {
  const pending: Array<{ value: unknown; exit?: boolean }> = [{ value }];
  const active = new WeakSet<object>();
  const visited = new WeakSet<object>();
  while (pending.length > 0) {
    const frame = pending.pop()!;
    const current = frame.value;
    const currentType = typeof current;
    if (currentType === 'string') {
      if (apiKey && (current as string).includes(apiKey)) return true;
      continue;
    }
    if (current === null || currentType === 'boolean') continue;
    if (currentType === 'number') {
      if (!Number.isFinite(current)) return true;
      continue;
    }
    if (
      currentType === 'undefined' ||
      currentType === 'bigint' ||
      currentType === 'symbol' ||
      currentType === 'function'
    ) {
      return true;
    }
    if (currentType !== 'object') return true;
    const objectValue = current as object;
    if (frame.exit) {
      active.delete(objectValue);
      visited.add(objectValue);
      continue;
    }
    if (active.has(objectValue)) return true;
    if (visited.has(objectValue)) continue;
    active.add(objectValue);
    pending.push({ value: objectValue, exit: true });
    try {
      const prototype = Object.getPrototypeOf(objectValue);
      if (!Array.isArray(objectValue) && prototype !== Object.prototype && prototype !== null) {
        return true;
      }
      for (const key of Reflect.ownKeys(objectValue)) {
        if (typeof key === 'symbol') return true;
        if (apiKey && key.includes(apiKey)) return true;
        const descriptor = Object.getOwnPropertyDescriptor(objectValue, key);
        if (!descriptor || 'get' in descriptor || 'set' in descriptor) return true;
        pending.push({ value: descriptor.value });
      }
    } catch {
      return true;
    }
  }
  return false;
}

/** Successful SDK payloads must be inert data, never executable accessors. */
function hasUnsafeAccessors(value: unknown, seen = new WeakSet<object>()): boolean {
  if (value === null || typeof value !== 'object') return false;
  if (seen.has(value)) return false;
  seen.add(value);
  try {
    for (const key of Reflect.ownKeys(value)) {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (!descriptor || 'get' in descriptor || 'set' in descriptor) return true;
      if (hasUnsafeAccessors(descriptor.value, seen)) return true;
    }
  } catch {
    return true;
  }
  return false;
}

type PlainDataSnapshot = { ok: true; value: unknown } | { ok: false };

/**
 * Copy an SDK response exclusively from own data descriptors. This prevents a
 * Proxy `get` trap from running after validation while retaining ordinary JSON
 * objects, arrays, and shared references.
 */
function snapshotPlainData(value: unknown): PlainDataSnapshot {
  const copies = new WeakMap<object, object>();
  const active = new WeakSet<object>();

  const copy = (current: unknown): PlainDataSnapshot => {
    if (
      current === null ||
      typeof current === 'string' ||
      typeof current === 'boolean' ||
      (typeof current === 'number' && Number.isFinite(current))
    ) return { ok: true, value: current };
    if (typeof current !== 'object') return { ok: false };

    const source = current as object;
    if (active.has(source)) return { ok: false };
    const existing = copies.get(source);
    if (existing) return { ok: true, value: existing };

    try {
      const isArray = Array.isArray(source);
      const prototype = Object.getPrototypeOf(source);
      if (!isArray && prototype !== Object.prototype && prototype !== null) return { ok: false };
      const ownKeys = Reflect.ownKeys(source);
      let arrayLength: number | undefined;
      if (isArray) {
        const lengthDescriptor = Object.getOwnPropertyDescriptor(source, 'length');
        if (
          !lengthDescriptor ||
          !('value' in lengthDescriptor) ||
          !Number.isSafeInteger(lengthDescriptor.value) ||
          lengthDescriptor.value < 0
        ) return { ok: false };
        arrayLength = lengthDescriptor.value;
        const indexKeys = ownKeys.filter((key) => key !== 'length');
        if (indexKeys.length !== arrayLength) return { ok: false };
        const indexes = new Set<number>();
        for (const key of indexKeys) {
          if (typeof key !== 'string') return { ok: false };
          const index = Number(key);
          if (!Number.isInteger(index) || index < 0 || index >= arrayLength || String(index) !== key) {
            return { ok: false };
          }
          indexes.add(index);
        }
        if (indexes.size !== arrayLength) return { ok: false };
      }
      const target: unknown[] | Record<string, unknown> = isArray ? [] : Object.create(null);
      copies.set(source, target);
      active.add(source);
      for (const key of ownKeys) {
        if (isArray && key === 'length') {
          Object.defineProperty(target, key, { value: arrayLength, writable: true });
          continue;
        }
        if (typeof key !== 'string') return { ok: false };
        const descriptor = Object.getOwnPropertyDescriptor(source, key);
        if (!descriptor || !('value' in descriptor) || !descriptor.enumerable) return { ok: false };
        const child = copy(descriptor.value);
        if (!child.ok) return child;
        Object.defineProperty(target, key, {
          value: child.value,
          enumerable: true,
          configurable: true,
          writable: true,
        });
      }
      active.delete(source);
      return { ok: true, value: target };
    } catch {
      return { ok: false };
    }
  };

  return copy(value);
}

function safeErrorReason(value: unknown): NimbleAgentRunErrorReason {
  return value === 'failed' || value === 'cancelled' || value === 'protocol' || value === 'request'
    ? value
    : 'request';
}

function safeCreateOutcome(value: unknown): NimbleAgentCreateOutcome | undefined {
  return value === 'unknown' || value === 'not-created' ? value : undefined;
}

function safeCreateErrorRunId(
  err: NimbleAgentRunError,
  agentId: string,
  apiKey: string | undefined,
): string | undefined {
  const runId = safeErrorMetadata(safeErrorProperty(err, 'runId'), apiKey);
  let returnedAgentId: unknown;
  try {
    returnedAgentId = err.agentId;
  } catch {
    return undefined;
  }
  return runId && /^task_run_[A-Za-z0-9_-]+$/.test(runId) &&
    (returnedAgentId === undefined ||
      (typeof returnedAgentId === 'string' &&
        safeErrorMetadata(returnedAgentId, apiKey) === agentId))
    ? runId
    : undefined;
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
  const nimbleError = isNimbleRunError(err);
  if (nimbleError) {
    const runRef = context.runId ? ` (run ${context.runId})` : '';
    const status = safeErrorProperty(err, 'status');
    const detail = context.allowErrorDetails
      ? safeErrorMessage(err, context.apiKey)
      : 'details withheld because the injected client credential was not provided for redaction';
    const message = `Nimble agent ${context.verb} failed${runRef}: ${detail}`;
    return new NimbleAgentRunError(message, {
      reason: safeErrorReason(safeErrorProperty(err, 'reason')),
      // Requested identifiers are trusted and authoritative on read paths.
      // Never copy model-visible metadata from an injected error when its
      // credential is unavailable for redaction.
      runId: context.runId,
      agentId: context.agentId,
      runStatus: context.allowErrorDetails
        ? safeErrorMetadata(safeErrorProperty(err, 'runStatus'), context.apiKey)
        : undefined,
      status: typeof status === 'number' ? status : undefined,
      createOutcome: safeCreateOutcome(safeErrorProperty(err, 'createOutcome')),
      cause: context.allowErrorDetails
        ? sanitizeCause(safeErrorProperty(err, 'cause'), context.apiKey)
        : undefined,
    });
  }
  const message = context.allowErrorDetails
    ? safeErrorMessage(err, context.apiKey)
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
function classifyCreateOutcome(status: number | undefined): NimbleAgentCreateOutcome {
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
  const nimbleError = isNimbleRunError(err);
  const status = readStatus(err);
  const suppliedOutcome = nimbleError
    ? safeCreateOutcome(safeErrorProperty(err, 'createOutcome'))
    : undefined;
  const classifiedOutcome = classifyCreateOutcome(status);
  const outcome = classifiedOutcome === 'unknown'
    ? 'unknown'
    : suppliedOutcome ?? classifiedOutcome;
  const message = context.allowErrorDetails
    ? safeErrorMessage(err, context.apiKey)
    : 'request failed; details withheld because the injected client credential was not provided for redaction';
  const guidance =
    outcome === 'not-created'
      ? 'No run was created; a retry only makes sense after the underlying condition is resolved.'
      : 'The run may or may not have been created server-side. Do not automatically start ' +
        'another run — list recent runs for this agent (or check the Nimble console) to ' +
        'reconcile before creating again.';
  const recoveredRunId =
    context.allowErrorDetails && nimbleError
      ? safeCreateErrorRunId(err, context.agentId, context.apiKey)
      : undefined;
  const runRef = recoveredRunId ? ` (run ${recoveredRunId})` : '';
  return new NimbleAgentRunError(`Nimble agent run creation failed${runRef}: ${message} ${guidance}`, {
    reason: nimbleError
      ? safeErrorReason(safeErrorProperty(err, 'reason'))
      : 'request',
    runId: recoveredRunId,
    agentId: context.agentId,
    runStatus:
      context.allowErrorDetails && nimbleError
        ? safeErrorMetadata(safeErrorProperty(err, 'runStatus'), context.apiKey)
        : undefined,
    status,
    createOutcome: outcome,
    cause: context.allowErrorDetails
      ? sanitizeCause(
          nimbleError ? safeErrorProperty(err, 'cause') : err,
          context.apiKey,
        )
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
  if (!/^wsa_[A-Za-z0-9_-]+$/.test(agentId)) {
    throw new NimbleConfigError('Invalid Nimble agent id: expected a wsa_ identifier.');
  }
  if (config.client) {
    // Only an explicitly paired key can be assumed to belong to an injected
    // client. An ambient NIMBLE_API_KEY may describe a different client and
    // therefore cannot make unknown error details safe to expose.
    const scrubKey = config.apiKey;
    if (scrubKey && agentId.includes(scrubKey)) {
      throw new NimbleConfigError('Nimble agent id contains protected credential material.');
    }
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
  if (agentId.includes(apiKey)) {
    throw new NimbleConfigError('Nimble agent id contains protected credential material.');
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

function hasValidRunFields(
  run: Partial<NimbleAgentRawRun>,
  apiKey: string | undefined,
): boolean {
  const error = run.error;
  const status = run.status;
  const containsApiKey =
    apiKey !== undefined &&
    [
      run.id,
      run.web_search_agent_id,
      run.interaction_id,
      run.status,
      run.effort,
      run.created_at,
      run.started_at,
      run.completed_at,
      run.prompt,
    ].some((value) => typeof value === 'string' && value.includes(apiKey));
  return (
    typeof run.id === 'string' &&
    run.id.length > 0 &&
    typeof run.web_search_agent_id === 'string' &&
    run.web_search_agent_id.length > 0 &&
    typeof run.interaction_id === 'string' &&
    run.interaction_id.length > 0 &&
    typeof run.is_active === 'boolean' &&
    LIFECYCLE_STATUSES.has(status as NimbleAgentRunLifecycleStatus) &&
    run.is_active === (status === 'queued' || status === 'running') &&
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
        typeof error.ref_id === 'string')) &&
    !containsApiKey
  );
}

/** Fail closed when a read/result body does not belong to the requested run. */
function assertMatchingRunIds(
  run: unknown,
  ids: { runId: string; agentId: string },
  apiKey: string | undefined,
): asserts run is NimbleAgentRawRun {
  if (typeof run !== 'object' || run === null) {
    throw protocolError(ids);
  }
  if (hasUnsafeAccessors(run)) throw protocolError(ids);
  const candidate = run as Partial<NimbleAgentRawRun>;
  if (candidate.id !== ids.runId || candidate.web_search_agent_id !== ids.agentId) {
    throw new NimbleAgentRunError(
      `Nimble agent run ${ids.runId} returned mismatched identifiers.`,
      { reason: 'protocol', runId: ids.runId, agentId: ids.agentId },
    );
  }
  if (!hasValidRunFields(candidate, apiKey)) {
    throw protocolError(ids);
  }
}

function snapshotRun(
  run: unknown,
  ids: { runId: string; agentId: string },
  apiKey: string | undefined,
): NimbleAgentRawRun {
  const snapshot = snapshotPlainData(run);
  if (!snapshot.ok) throw protocolError(ids);
  assertMatchingRunIds(snapshot.value, ids, apiKey);
  return snapshot.value;
}

function createProtocolError(agentId: string, runId?: string): NimbleAgentRunError {
  return new NimbleAgentRunError(
    'Nimble agent run creation returned a malformed payload after the request was accepted. ' +
      (runId
        ? `Run ${runId} may have been created server-side; resume it or reconcile recent runs before creating again.`
        : 'The run may have been created server-side; reconcile recent runs before creating again.'),
    {
      reason: 'protocol',
      ...(runId ? { runId } : {}),
      agentId,
      createOutcome: 'unknown',
    },
  );
}

function safeCreatedRunId(
  run: unknown,
  agentId: string,
  apiKey: string | undefined,
): string | undefined {
  if (typeof run !== 'object' || run === null) return undefined;
  const candidate = run as { id?: unknown; web_search_agent_id?: unknown };
  const id = candidate.id;
  return typeof id === 'string' &&
    /^task_run_[A-Za-z0-9_-]+$/.test(id) &&
    candidate.web_search_agent_id === agentId &&
    !containsCredential(id, apiKey)
    ? id
    : undefined;
}

function snapshotCreatedRun(
  run: unknown,
  agentId: string,
  apiKey: string | undefined,
): NimbleAgentRawRun {
  const snapshot = snapshotPlainData(run);
  if (!snapshot.ok || typeof snapshot.value !== 'object' || snapshot.value === null) {
    throw createProtocolError(agentId);
  }
  const candidate = snapshot.value as NimbleAgentRawRun;
  if (
    !hasValidRunFields(candidate, apiKey) ||
    candidate.web_search_agent_id !== agentId ||
    !/^task_run_[A-Za-z0-9_-]+$/.test(candidate.id)
  ) {
    // The POST was accepted. Preserve a separately validated run handle even
    // when another response field is malformed, so callers can reconcile or
    // resume instead of risking a second billed create.
    throw createProtocolError(agentId, safeCreatedRunId(candidate, agentId, apiKey));
  }
  return candidate;
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

function toTerminalNotReadyOutput(run: NimbleAgentRawRun): NimbleAgentRunPendingOutput {
  return {
    ready: false,
    ...baseFields(run),
    status: run.status as 'completed' | 'failed' | 'cancelled',
    isActive: false,
    ...(run.started_at ? { startedAt: run.started_at } : {}),
    ...(run.completed_at ? { completedAt: run.completed_at } : {}),
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
    return { type: 'text', text: raw.content, trust: raw.trust as NimbleAgentTrust };
  }
  if (kind === 'json' && typeof raw.content === 'object' && raw.content !== null) {
    return { type: 'json', json: raw.content, trust: raw.trust as NimbleAgentTrust };
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

/** Await an operation without trusting the implementation to honor its signal. */
function abortable<T>(operation: PromiseLike<T>, signal: AbortSignal | undefined): Promise<T> {
  const promise = Promise.resolve(operation);
  if (!signal) return promise;
  if (signal.aborted) {
    void promise.catch(() => undefined);
    return Promise.reject(abortReason(signal));
  }
  return new Promise<T>((resolve, reject) => {
    const settle = (callback: (value: never) => void, value: unknown) => {
      signal.removeEventListener('abort', onAbort);
      callback(value as never);
    };
    const onAbort = () => settle(reject, abortReason(signal));
    signal.addEventListener('abort', onAbort, { once: true });
    promise.then(
      (value) => settle(resolve, value),
      (error) => settle(reject, error),
    );
  });
}

/**
 * A finite, positive number, or the fallback. Guards against non-finite /
 * non-positive wait values (notably `NaN` from `Number(unset env)`) — which
 * `??` would treat as "provided", leaving a `NaN` timeout that never trips the
 * `remaining <= 0` break and a `NaN` sleep that coerces to a 0ms tight poll.
 */
function positiveDuration(value: number | undefined, fallback: number): number {
  const duration =
    typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback;
  // AbortSignal.timeout requires an integer; this bound also avoids setTimeout's
  // platform overflow behavior for very large delays.
  return Math.min(Math.ceil(duration), 2_147_483_647);
}

function normalizeWait(
  wait: NimbleAgentRunResultConfig['wait'],
): Required<NimbleAgentWaitOptions> | undefined {
  if (!wait) return undefined;
  const options = wait === true ? {} : wait;
  return {
    timeoutMs: positiveDuration(options.timeoutMs, NIMBLE_AGENT_DEFAULTS.waitTimeoutMs),
    pollIntervalMs: Math.max(
      positiveDuration(options.pollIntervalMs, NIMBLE_AGENT_DEFAULTS.pollIntervalMs),
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
      if (
        config.sources &&
        (config.sources.allow.length === 0 ||
          config.sources.allow.some(
            (group) => !group.title.trim() || group.domains.length === 0 || group.domains.some((d) => !d.trim()),
          ))
      ) {
        throw new NimbleConfigError(
          'Nimble Agent API sources.allow must contain at least one named source group with non-empty domains.',
        );
      }
      const body: NimbleAgentRunCreateBody = {
        input: input.task,
        ...(effort ? { effort } : {}),
        ...(config.outputSchema ? { output_schema: config.outputSchema } : {}),
        ...(config.sources ? { sources: config.sources } : {}),
      };

      let run: NimbleAgentRawRun;
      try {
        if (signal?.aborted) throw abortReason(signal);
        // One-shot by contract: never let the SDK's default retry policy
        // (408/409/429/5xx) replay a non-idempotent, billed POST.
        run = await abortable(
          client.agents.runs.create(agentId, body, {
            ...requestOptions(signal),
            maxRetries: 0,
          }),
          signal,
        );
      } catch (err) {
        throw toCreateError(err, { agentId, apiKey, allowErrorDetails });
      }
      return toStartOutput(snapshotCreatedRun(run, agentId, apiKey));
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
      assertSafeRequestedRunId(input.runId, agentId, apiKey);

      let run: NimbleAgentRawRun;
      try {
        run = await abortable(
          client.agents.runs.get(
            input.runId,
            { agent_id: agentId },
            requestOptions(signal),
          ),
          signal,
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
      run = snapshotRun(run, { runId: input.runId, agentId }, apiKey);
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
      assertSafeRequestedRunId(input.runId, agentId, apiKey);
      const ids = { runId: input.runId, agentId };

      const getRun = async (requestSignal = signal): Promise<NimbleAgentRawRun> => {
        try {
          const fetched = await abortable(
            client.agents.runs.get(
              input.runId,
              { agent_id: agentId },
              requestOptions(requestSignal),
            ),
            requestSignal,
          );
          return snapshotRun(fetched, ids, apiKey);
        } catch (err) {
          throw toAgentError(err, {
            verb: 'status check',
            ...ids,
            apiKey,
            allowErrorDetails,
          });
        }
      };

      const sleepBeforePoll = async (ms: number): Promise<void> => {
        try {
          await sleep(ms, signal);
        } catch (err) {
          throw toAgentError(err, {
            verb: 'wait',
            ...ids,
            apiKey,
            allowErrorDetails,
          });
        }
      };

      const startedWaiting = wait ? performance.now() : undefined;
      const waitExpired = () =>
        wait !== undefined && performance.now() - startedWaiting! >= wait.timeoutMs;
      const initialDeadlineSignal = wait
        ? AbortSignal.timeout(wait.timeoutMs)
        : undefined;
      const initialSignal = initialDeadlineSignal
        ? signal
          ? AbortSignal.any([signal, initialDeadlineSignal])
          : initialDeadlineSignal
        : signal;
      let run: NimbleAgentRawRun;
      try {
        run = await getRun(initialSignal);
      } catch (err) {
        if ((initialDeadlineSignal?.aborted || waitExpired()) && !signal?.aborted) {
          return {
            ready: false,
            runId: ids.runId,
            agentId: ids.agentId,
            status: 'unknown',
          };
        }
        throw err;
      }
      assertKnownStatus(run, ids);
      if (waitExpired()) {
        if (run.status === 'completed' || run.status === 'failed' || run.status === 'cancelled') {
          return toTerminalNotReadyOutput(run);
        }
        if (run.status === 'queued' || run.status === 'running') {
          return toPendingOutput(run, run.status);
        }
      }

      if (wait && run.is_active) {
        // Established before the initial status request, so timeoutMs bounds
        // the complete wait operation rather than only follow-up polls.
        const waitStartedAt = startedWaiting!;
        while (run.is_active) {
          const elapsed = performance.now() - waitStartedAt;
          const remaining = wait.timeoutMs - elapsed;
          if (remaining <= 0) break;
          if (remaining <= wait.pollIntervalMs) {
            await sleepBeforePoll(remaining);
            break;
          }
          await sleepBeforePoll(wait.pollIntervalMs);
          const remainingAfterSleep = wait.timeoutMs - (performance.now() - waitStartedAt);
          if (remainingAfterSleep <= 0) break;
          const deadlineSignal = AbortSignal.timeout(Math.max(1, Math.ceil(remainingAfterSleep)));
          const requestSignal = signal
            ? AbortSignal.any([signal, deadlineSignal])
            : deadlineSignal;
          let fetched: NimbleAgentRawRun;
          try {
            fetched = await getRun(requestSignal);
          } catch (err) {
            if ((deadlineSignal.aborted || waitExpired()) && !signal?.aborted) break;
            throw err;
          }
          if (performance.now() - waitStartedAt >= wait.timeoutMs) {
            assertKnownStatus(fetched, ids);
            if (signal?.aborted) {
              throw toAgentError(signal.reason, {
                verb: 'status check',
                ...ids,
                apiKey,
                allowErrorDetails,
              });
            }
            if (
              fetched.status === 'completed' ||
              fetched.status === 'failed' ||
              fetched.status === 'cancelled'
            ) {
              return toTerminalNotReadyOutput(fetched);
            }
            break;
          }
          run = fetched;
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
      while (true) {
        try {
          result = await abortable(
            client.agents.runs.result(
              input.runId,
              { agent_id: agentId },
              requestOptions(wait ? initialSignal : signal),
            ),
            wait ? initialSignal : signal,
          );
          break;
        } catch (err) {
          if (wait && (initialDeadlineSignal?.aborted || waitExpired()) && !signal?.aborted) {
            return toTerminalNotReadyOutput(run);
          }
          const httpStatus = readStatus(err);
          // 409: status/result eventual consistency. A bounded waiter keeps
          // trying within the original deadline; an unbounded call returns.
          if (httpStatus === 409) {
            if (!wait) {
              return toPendingOutput({ ...run, status: 'running', is_active: true }, 'running');
            }
            const remaining = wait.timeoutMs - (performance.now() - startedWaiting!);
            if (remaining <= 0) return toTerminalNotReadyOutput(run);
            if (remaining <= wait.pollIntervalMs) {
              await sleepBeforePoll(remaining);
              return toTerminalNotReadyOutput(run);
            }
            await sleepBeforePoll(wait.pollIntervalMs);
            if (waitExpired()) return toTerminalNotReadyOutput(run);
            continue;
          }
        // 422: terminal failure — the body carries the run + structured error.
        if (httpStatus === 422) {
          const failed = readFailedResultBody(err);
          if (failed) {
            assertMatchingRunIds(failed.run, ids, apiKey);
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
          throw protocolError(ids);
        }
          throw toAgentError(err, {
            verb: 'result fetch',
            ...ids,
            apiKey,
            allowErrorDetails,
          });
        }
      }

      // Enforce cancellation independently of client settlement behavior. An
      // injected client may ignore the signal and resolve after it fires.
      if (signal?.aborted) {
        throw toAgentError(signal.reason, {
          verb: 'result fetch',
          ...ids,
          apiKey,
          allowErrorDetails,
        });
      }
      if (wait && (initialDeadlineSignal?.aborted || waitExpired())) {
        return toTerminalNotReadyOutput(run);
      }

      const resultSnapshot = snapshotPlainData(result);
      if (!resultSnapshot.ok || typeof resultSnapshot.value !== 'object' || resultSnapshot.value === null) {
        throw protocolError(ids);
      }
      result = resultSnapshot.value as NimbleAgentRawResult | NimbleAgentRawFailedResult;
      // A successful result is model-visible. Reject the complete envelope if
      // a backend/proxy reflects the configured server-only credential in the
      // answer, structured JSON, trust metadata, or any future field.
      if (containsCredential(result, apiKey)) {
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
        assertMatchingRunIds(failed.run, ids, apiKey);
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
      assertMatchingRunIds(resultRun, ids, apiKey);
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
