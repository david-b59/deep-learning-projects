import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // Alias « @ » = dossier src (imports courts, utilisés par shadcn/ui)
    alias: { '@': path.resolve(import.meta.dirname, './src') },
  },
})
