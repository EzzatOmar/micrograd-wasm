import { defineConfig } from 'vite'
import wasm from "vite-plugin-wasm";
import tla from "vite-plugin-top-level-await";

export default defineConfig({
    plugins: [wasm(), tla()],
    server: {
        open: false
    }
})
