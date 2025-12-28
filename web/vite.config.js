import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8088",
      "/reports": "http://localhost:8088",
      "/models": "http://localhost:8088"
    }
  }
});
