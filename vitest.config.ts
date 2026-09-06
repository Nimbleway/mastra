import { defineConfig } from 'vitest/config';

// The default `vitest run` covers the mocked unit suite only — no network, no
// NIMBLE_API_KEY. Live validation is a separate, operator-run concern.
export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
    globals: false,
  },
});
