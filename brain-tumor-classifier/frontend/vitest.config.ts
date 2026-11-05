/**
 * Configuration minimale de Vitest : les tests portent sur des fonctions pures, l'environnement
 * Node suffit (pas de navigateur simulé).
 */
import path from 'node:path'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  resolve: {
    // Même alias « @ » que pour le site
    alias: { '@': path.resolve(import.meta.dirname, './src') },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
