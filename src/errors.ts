/**
 * Thrown when a tool is invoked without a resolvable API key or agent id.
 * Raised at execute time, not factory time, so tools can be constructed in
 * key-less environments (unit tests, type-checking, build).
 */
export class NimbleConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'NimbleConfigError';
  }
}

/** Why an agent-run tool call failed. */
export type NimbleAgentRunErrorReason =
  /** The run reached terminal status `failed`. */
  | 'failed'
  /** The run reached terminal status `cancelled`. */
  | 'cancelled'
  /** The API returned something outside the documented contract. */
  | 'protocol'
  /** The underlying request errored (transport, auth, rate limit, …). */
  | 'request';

/**
 * Whether a failed run-creation request actually created a run server-side.
 * `unknown` means the request may have reached the server (timeout, 408, 409,
 * 5xx, connection drop) — never blindly re-create; inspect existing runs
 * first. `not-created` means the server definitively rejected it (4xx like 429).
 */
export type NimbleAgentCreateOutcome = 'unknown' | 'not-created';

/**
 * Thrown by the agent tools when a run cannot produce a result. Always retains
 * the runId (when known) in the fields *and* the message, so a model or caller
 * seeing the error can still resume, inspect, or report the run.
 */
export class NimbleAgentRunError extends Error {
  /** The run this error belongs to, when known. */
  readonly runId?: string;
  /** The agent instance the run belongs to, when known. */
  readonly agentId?: string;
  /** The run's terminal lifecycle status, when the server reported one. */
  readonly runStatus?: string;
  readonly reason: NimbleAgentRunErrorReason;
  /** HTTP status of the underlying request, when available. */
  readonly status?: number;
  /** For failed run creation only: did a run get created server-side? */
  readonly createOutcome?: NimbleAgentCreateOutcome;

  constructor(
    message: string,
    options: {
      reason: NimbleAgentRunErrorReason;
      runId?: string;
      agentId?: string;
      runStatus?: string;
      status?: number;
      createOutcome?: NimbleAgentCreateOutcome;
      cause?: unknown;
    },
  ) {
    super(message, options.cause !== undefined ? { cause: options.cause } : undefined);
    this.name = 'NimbleAgentRunError';
    this.reason = options.reason;
    this.runId = options.runId;
    this.agentId = options.agentId;
    this.runStatus = options.runStatus;
    this.status = options.status;
    this.createOutcome = options.createOutcome;
  }
}
