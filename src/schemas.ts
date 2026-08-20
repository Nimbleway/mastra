import { z } from 'zod';
import type { NimbleClientOptions } from './client';

// ── Model-facing input schemas ─────────────────────────────────────────────

/**
 * Effort tiers accepted by the Agent API, in ascending cost/latency order.
 * Higher tiers research more sources for longer before answering.
 */
export const NIMBLE_AGENT_EFFORTS = ['low', 'medium', 'high', 'x-high', 'max'] as const;

export type NimbleAgentEffort = (typeof NIMBLE_AGENT_EFFORTS)[number];

/**
 * Input for the start-run tool. The model chooses the research task and, at
 * most, an effort tier (clamped to the developer-configured `effortCap`).
 * Everything else — the agent instance, credentials, request policy — is
 * developer configuration and never appears in the model schema.
 */
export const nimbleAgentStartRunInputSchema = z.object({
  task: z
    .string()
    .min(1)
    .describe('The research task or question for the Nimble deep-research agent.'),
  effort: z
    .enum(NIMBLE_AGENT_EFFORTS)
    .optional()
    .describe(
      'Optional effort tier. Higher tiers research more sources and take ' +
        'longer (minutes) and cost more; omit to use the configured default.',
    ),
});

export type NimbleAgentStartRunInput = z.infer<typeof nimbleAgentStartRunInputSchema>;

/**
 * Input for the status and result tools: just the run ID returned by the
 * start-run tool. Runs are resumable — any process configured with the same
 * agent can check a run it did not start.
 */
export const nimbleAgentRunIdInputSchema = z.object({
  runId: z
    .string()
    .min(1)
    .describe(
      'The Nimble agent run ID (format "task_run_<uuid>") returned when the run was started.',
    ),
});

export type NimbleAgentRunIdInput = z.infer<typeof nimbleAgentRunIdInputSchema>;

// ── Lifecycle ──────────────────────────────────────────────────────────────

/** Run lifecycle states. `queued` and `running` are the non-terminal pair. */
export const NIMBLE_AGENT_RUN_STATUSES = [
  'queued',
  'running',
  'completed',
  'failed',
  'cancelled',
] as const;

export type NimbleAgentRunLifecycleStatus = (typeof NIMBLE_AGENT_RUN_STATUSES)[number];

// ── Trust metadata (zod, passthrough so future fields survive verbatim) ────

const trustConfidenceSchema = z.enum(['high', 'medium', 'low', 'pre_existing']);
const sourceTypeSchema = z.enum(['primary', 'secondary']);
const sourceCategorySchema = z.enum([
  'official',
  'news',
  'social',
  'academic',
  'aggregator',
  'other',
]);

/** A source consulted while producing the answer. */
export const nimbleAgentTrustSourceSchema = z
  .object({
    url: z.string(),
    type: sourceTypeSchema,
    title: z.string().nullish(),
    source_category: sourceCategorySchema.nullish(),
    source_intent: sourceCategorySchema.nullish(),
    extract_template_name: z.string().nullish(),
  })
  .passthrough();

/** A citation backing a specific claim, with verbatim supporting excerpts. */
export const nimbleAgentTrustCitationSchema = z
  .object({
    url: z.string(),
    title: z.string().nullish(),
    excerpts: z.array(z.string()).nullish(),
    source_type: sourceTypeSchema.nullish(),
    source_category: sourceCategorySchema.nullish(),
    source_intent: sourceCategorySchema.nullish(),
    extract_template_name: z.string().nullish(),
  })
  .passthrough();

/**
 * Trust metadata for one claim. Text answers key claims by `callout` (numeric
 * markers embedded in the prose); JSON answers key claims by `path` (the JSON
 * path of the value).
 */
export const nimbleAgentTrustClaimSchema = z
  .object({
    confidence: trustConfidenceSchema,
    reasoning: z.string(),
    citations: z.array(nimbleAgentTrustCitationSchema),
    callout: z.number().optional(),
    path: z.string().optional(),
  })
  .passthrough();

/**
 * Trust and citation metadata for a run's output — passed through verbatim
 * from the API (snake_case preserved) so citation markers stay aligned with
 * the answer and future fields survive.
 */
export const nimbleAgentTrustSchema = z
  .object({
    confidence: trustConfidenceSchema,
    reasoning: z.string(),
    sources: z.array(nimbleAgentTrustSourceSchema),
    claims: z.array(nimbleAgentTrustClaimSchema),
  })
  .passthrough();

export type NimbleAgentTrust = z.infer<typeof nimbleAgentTrustSchema>;

// ── Tool output schemas ────────────────────────────────────────────────────

const effortSchema = z.enum(NIMBLE_AGENT_EFFORTS);
const lifecycleStatusSchema = z.enum(NIMBLE_AGENT_RUN_STATUSES);

/** Output of the start-run tool: the handle needed to resume later. */
export const nimbleAgentStartRunOutputSchema = z.object({
  /** The real run ID (`task_run_<uuid>`) — pass to the status/result tools. */
  runId: z.string(),
  /** The agent instance the run belongs to (`web_search_agent_id`). */
  agentId: z.string(),
  /** Interaction ID (conversation-continuation handle). */
  interactionId: z.string(),
  status: lifecycleStatusSchema,
  effort: effortSchema,
  createdAt: z.string(),
});

export type NimbleAgentStartRunOutput = z.infer<typeof nimbleAgentStartRunOutputSchema>;

/** Output of the status tool: a point-in-time run snapshot. */
export const nimbleAgentRunStatusOutputSchema = z.object({
  runId: z.string(),
  agentId: z.string(),
  status: lifecycleStatusSchema,
  /** True while the run is still queued or running. */
  isActive: z.boolean(),
  effort: effortSchema,
  createdAt: z.string(),
  startedAt: z.string().optional(),
  completedAt: z.string().optional(),
  /** Server-reported error details when the run failed. */
  error: z.object({ message: z.string() }).optional(),
});

export type NimbleAgentRunStatusOutput = z.infer<typeof nimbleAgentRunStatusOutputSchema>;

/** Result-tool output while the run is still working: try again later. */
export const nimbleAgentRunPendingOutputSchema = z.object({
  ready: z.literal(false),
  runId: z.string(),
  agentId: z.string(),
  status: z.enum(['queued', 'running']),
  isActive: z.literal(true),
  effort: effortSchema,
  createdAt: z.string(),
  startedAt: z.string().optional(),
});

export type NimbleAgentRunPendingOutput = z.infer<typeof nimbleAgentRunPendingOutputSchema>;

/** The completed run's answer: prose (`text`) or structured (`json`). */
export const nimbleAgentOutputSchema = z.union([
  z.object({ type: z.literal('text'), text: z.string(), trust: nimbleAgentTrustSchema }),
  z.object({
    type: z.literal('json'),
    json: z.union([z.record(z.string(), z.unknown()), z.array(z.unknown())]),
    trust: nimbleAgentTrustSchema,
  }),
]);

export type NimbleAgentOutput = z.infer<typeof nimbleAgentOutputSchema>;

/** Result-tool output once the run completed. */
export const nimbleAgentRunCompletedOutputSchema = z.object({
  ready: z.literal(true),
  runId: z.string(),
  agentId: z.string(),
  status: z.literal('completed'),
  effort: effortSchema,
  createdAt: z.string(),
  startedAt: z.string().optional(),
  completedAt: z.string().optional(),
  output: nimbleAgentOutputSchema,
});

export type NimbleAgentRunCompletedOutput = z.infer<typeof nimbleAgentRunCompletedOutputSchema>;

export const nimbleAgentRunResultOutputSchema = z.union([
  nimbleAgentRunPendingOutputSchema,
  nimbleAgentRunCompletedOutputSchema,
]);

export type NimbleAgentRunResultOutput = z.infer<typeof nimbleAgentRunResultOutputSchema>;

// ── Developer-facing factory configs ───────────────────────────────────────

/** Config shared by all three agent tool factories. */
export interface NimbleAgentToolConfig {
  /**
   * The Web Search Agent instance to run (format `wsa_<uuid>`), created once
   * via the Nimble console or API. Defaults to `process.env.NIMBLE_AGENT_ID`.
   * Resolved at execute time; the model can never choose the agent.
   */
  agentId?: string;
  /** Nimble API key. Defaults to `process.env.NIMBLE_API_KEY` (server-side). */
  apiKey?: string;
  /** Inject a pre-built / mock client (tests, advanced users). When set, the
   * package does not construct a client and `clientOptions` is ignored;
   * attribution headers are the injected client's concern. Also provide
   * the matching `apiKey` if server error details should be preserved:
   * without the credential, those details and raw causes are withheld because
   * the package cannot prove that they are safe to expose. */
  client?: NimbleAgentRunsClient;
  /** Options forwarded to the package-constructed Nimble client. */
  clientOptions?: NimbleClientOptions;
}

/** Config for the start-run tool factory. */
export interface NimbleAgentStartRunConfig extends NimbleAgentToolConfig {
  /**
   * Effort used when the model does not choose one. Unset means the agent
   * instance's own configured default applies (recommended).
   */
  effort?: NimbleAgentEffort;
  /**
   * Upper bound on the effort the *model* may request; model choices above it
   * are clamped. Does not limit the developer-set `effort`. Default `high`,
   * so a model cannot unilaterally trigger `x-high`/`max` cost tiers.
   */
  effortCap?: NimbleAgentEffort;
}

/** Bounded-wait behavior for the result tool. */
export interface NimbleAgentWaitOptions {
  /** Give up waiting after this long (run keeps going server-side). Default 300_000. */
  timeoutMs?: number;
  /** Delay between status polls. Default 10_000, floor 100. */
  pollIntervalMs?: number;
}

/** Config for the result tool factory. */
export interface NimbleAgentRunResultConfig extends NimbleAgentToolConfig {
  /**
   * When set, the tool polls a still-active run until it finishes or the
   * bounded timeout elapses (`true` = defaults). When absent (the default),
   * the tool never blocks: an active run returns `{ ready: false }`
   * immediately. Waiting respects Mastra's per-call `AbortSignal`.
   */
  wait?: boolean | NimbleAgentWaitOptions;
}

/** Config for the convenience factory: everything, shared across all tools. */
export interface NimbleAgentToolsConfig extends NimbleAgentStartRunConfig {
  wait?: boolean | NimbleAgentWaitOptions;
}

// ── Structural surface of the SDK this package calls ───────────────────────
// Declared structurally (mirroring `@nimble-way/nimble-js@1.2.x` generated
// types) so tests can inject mocks and the package's public .d.ts stays
// self-contained.

/** Body this package sends to `POST /v2/agents/{agent_id}/runs`. */
export interface NimbleAgentRunCreateBody {
  /** User prompt or task instructions for the run. */
  input: string;
  /** Effort level overriding the agent default for this run. */
  effort?: NimbleAgentEffort | null;
}

/**
 * Per-call options this package forwards to the SDK: abort propagation, and —
 * on run creation only — `maxRetries: 0` so a non-idempotent POST is one-shot.
 */
export interface NimbleAgentRequestOptions {
  signal?: AbortSignal | undefined;
  maxRetries?: number;
}

/** Raw run object returned by create/get (TaskRunResponsePublicV2). */
export interface NimbleAgentRawRun {
  /** Run identifier, format `task_run_<uuid>`. */
  id: string;
  interaction_id: string;
  status: NimbleAgentRunLifecycleStatus;
  /** True while status is `queued` or `running`. */
  is_active: boolean;
  effort: NimbleAgentEffort;
  created_at: string;
  web_search_agent_id: string;
  started_at?: string | null;
  completed_at?: string | null;
  prompt?: string | null;
  error?: { message: string; ref_id: string } | null;
}

/** Raw output union returned by `GET .../result` on success. */
export type NimbleAgentRawOutput =
  | { type?: 'text'; content: string; trust: NimbleAgentTrust }
  | { type?: 'json'; content: Record<string, unknown> | unknown[]; trust: NimbleAgentTrust };

/** Raw success form of the result endpoint. */
export interface NimbleAgentRawResult {
  run: NimbleAgentRawRun;
  output: NimbleAgentRawOutput;
}

/** Raw failed form of the result endpoint (also carried on HTTP 422). */
export interface NimbleAgentRawFailedResult {
  run: NimbleAgentRawRun;
  error: { message: string; ref_id: string };
}

/**
 * The slice of `@nimble-way/nimble-js` the agent tools call:
 * `client.agents.runs.create/get/result`.
 */
export interface NimbleAgentRunsClient {
  agents: {
    runs: {
      create(
        agentId: string,
        body: NimbleAgentRunCreateBody,
        options?: NimbleAgentRequestOptions,
      ): Promise<NimbleAgentRawRun>;
      get(
        runId: string,
        params: { agent_id: string },
        options?: NimbleAgentRequestOptions,
      ): Promise<NimbleAgentRawRun>;
      result(
        runId: string,
        params: { agent_id: string },
        options?: NimbleAgentRequestOptions,
      ): Promise<NimbleAgentRawResult | NimbleAgentRawFailedResult>;
    };
  };
}
