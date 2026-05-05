import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";

export default defineConfig([
  ...nextVitals,
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    "src/App.jsx",
    "src/App.css",
    "src/main.jsx",
    "src/index.css",
    "src/assets/**",
    "vite.config.js",
    "index.html",
  ]),
]);
