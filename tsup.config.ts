import { defineConfig } from 'tsup';

// ESM-first dual build (ESM + CJS + .d.ts). Peer/runtime deps (`@mastra/core`,
// `zod`, `@nimble-way/nimble-js`) are auto-externalized by tsup from
// package.json, so the consumer's own copy is used — avoiding duplicate-zod
// tool-schema errors.
export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  outExtension: ({ format }) => ({ js: format === 'cjs' ? '.cjs' : '.js' }),
  dts: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  target: 'node22',
});
