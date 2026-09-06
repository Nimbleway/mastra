import { Nimble } from '@nimble-way/nimble-js';

/**
 * The attribution value this package stamps on every Nimble API request it
 * makes. The Nimble client sends it as the `X-Client-Source` header on every
 * request (via the SDK's `clientSource` option), letting Nimble attribute
 * traffic to this integration.
 */
export const NIMBLE_CLIENT_SOURCE = 'mastra';

/**
 * Options forwarded to the underlying `@nimble-way/nimble-js` client when a
 * tool constructs its own client (ignored when a pre-built `client` is
 * injected). A deliberate subset of the SDK's `ClientOptions` — enough for
 * proxies, custom runtimes, and request capture in tests — so this package's
 * public types stay self-contained.
 */
export interface NimbleClientOptions {
  /** Override the API base URL (e.g. a proxy or a mock server). */
  baseURL?: string;
  /** Custom `fetch` implementation (instrumentation, polyfills, tests). */
  fetch?: typeof fetch;
  /** Per-request timeout in milliseconds (SDK default applies when unset). */
  timeout?: number;
  /**
   * SDK retry count for *read* requests (status / result polls). The SDK
   * default (2) applies when unset. Run creation is always sent with a
   * per-request `maxRetries: 0` regardless of this value — a create is not
   * idempotent and must never be silently replayed.
   */
  maxRetries?: number;
}

/**
 * Build the Nimble client this package uses: the caller's key plus this
 * package's `X-Client-Source: mastra` attribution. Every tool that constructs
 * its own client goes through here so the attribution cannot drift.
 */
export function createNimbleClient(apiKey: string, options: NimbleClientOptions = {}): Nimble {
  return new Nimble({
    ...options,
    apiKey,
    clientSource: NIMBLE_CLIENT_SOURCE,
  });
}
