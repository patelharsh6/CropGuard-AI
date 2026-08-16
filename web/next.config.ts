import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* Disable browser console log forwarding to the dev terminal.
   * LiteRT.js's Emscripten WASM engine outputs verbose C++ init logs via
   * low-level _fd_write (bypassing JS console.*), which Next.js Turbopack
   * intercepts and prints with full stack traces, flooding the terminal.
   * These are harmless hardware-probe logs, not errors. */
  logging: {
    incomingRequests: true,
    fetches: {},
    browserConsole: false,
  },
  // Allow phones/tablets on the local network to access the dev server.
  // Next.js 16 blocks cross-origin dev resource requests by default.
  allowedDevOrigins: ['10.64.111.37'],
};

export default nextConfig;
