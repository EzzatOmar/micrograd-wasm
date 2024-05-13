import './style.css'
import typescriptLogo from './typescript.svg'
import viteLogo from '/vite.svg'
import {testing_sin} from "micrograd-wasm";

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div>
    <a href="https://vitejs.dev" target="_blank">
      <img src="${viteLogo}" class="logo" alt="Vite logo" />
    </a>
    <a href="https://www.typescriptlang.org/" target="_blank">
      <img src="${typescriptLogo}" class="logo vanilla" alt="TypeScript logo" />
    </a>
    <h1>Vite + TypeScript + Wasm + Rust</h1>
    <div class="card">
      <button id="counter" type="button">Start training [10k steps], check the console</button>
    </div>
    <p class="read-the-docs">
      We built a simple neural net in Rust and compiled it to WebAssembly.
      <br />
      The nn is approximating the sin function.
      <br />
      Check lib.rs and main.ts for more info.
    </p>
  </div>
`

export function setupWasm(element: HTMLButtonElement) {
  const setWasm = () => {
    testing_sin()
  }
  element.addEventListener('click', () => setWasm())
}

setupWasm(document.querySelector<HTMLButtonElement>('#counter')!)

//setupCounter(document.querySelector<HTMLButtonElement>('#counter')!)
