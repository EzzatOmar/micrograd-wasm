var bn=(m,_)=>()=>(_||m((_={exports:{}}).exports,_),_.exports);var dn=bn((wn,p)=>{(async()=>{(function(){const n=document.createElement("link").relList;if(n&&n.supports&&n.supports("modulepreload"))return;for(const e of document.querySelectorAll('link[rel="modulepreload"]'))o(e);new MutationObserver(e=>{for(const i of e)if(i.type==="childList")for(const u of i.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&o(u)}).observe(document,{childList:!0,subtree:!0});function t(e){const i={};return e.integrity&&(i.integrity=e.integrity),e.referrerPolicy&&(i.referrerPolicy=e.referrerPolicy),e.crossOrigin==="use-credentials"?i.credentials="include":e.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function o(e){if(e.ep)return;e.ep=!0;const i=t(e);fetch(e.href,i)}})();const m="data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20xmlns:xlink='http://www.w3.org/1999/xlink'%20aria-hidden='true'%20role='img'%20class='iconify%20iconify--logos'%20width='32'%20height='32'%20preserveAspectRatio='xMidYMid%20meet'%20viewBox='0%200%20256%20256'%3e%3cpath%20fill='%23007ACC'%20d='M0%20128v128h256V0H0z'%3e%3c/path%3e%3cpath%20fill='%23FFF'%20d='m56.612%20128.85l-.081%2010.483h33.32v94.68h23.568v-94.68h33.321v-10.28c0-5.69-.122-10.444-.284-10.566c-.122-.162-20.4-.244-44.983-.203l-44.74.122l-.121%2010.443Zm149.955-10.742c6.501%201.625%2011.459%204.51%2016.01%209.224c2.357%202.52%205.851%207.111%206.136%208.208c.08.325-11.053%207.802-17.798%2011.988c-.244.162-1.22-.894-2.317-2.52c-3.291-4.795-6.745-6.867-12.028-7.233c-7.76-.528-12.759%203.535-12.718%2010.321c0%201.992.284%203.17%201.097%204.795c1.707%203.536%204.876%205.649%2014.832%209.956c18.326%207.883%2026.168%2013.084%2031.045%2020.48c5.445%208.249%206.664%2021.415%202.966%2031.208c-4.063%2010.646-14.14%2017.879-28.323%2020.276c-4.388.772-14.79.65-19.504-.203c-10.28-1.828-20.033-6.908-26.047-13.572c-2.357-2.6-6.949-9.387-6.664-9.874c.122-.163%201.178-.813%202.356-1.504c1.138-.65%205.446-3.129%209.509-5.485l7.355-4.267l1.544%202.276c2.154%203.29%206.867%207.801%209.712%209.305c8.167%204.307%2019.383%203.698%2024.909-1.26c2.357-2.153%203.332-4.388%203.332-7.68c0-2.966-.366-4.266-1.91-6.501c-1.99-2.845-6.054-5.242-17.595-10.24c-13.206-5.69-18.895-9.224-24.096-14.832c-3.007-3.25-5.852-8.452-7.03-12.8c-.975-3.617-1.22-12.678-.447-16.335c2.723-12.76%2012.353-21.659%2026.25-24.3c4.51-.853%2014.994-.528%2019.424.569Z'%3e%3c/path%3e%3c/svg%3e",_="/vite.svg",x="/assets/micrograd_wasm_bg-C4nMy7EV.wasm",A=async(n={},t)=>{let o;if(t.startsWith("data:")){const e=t.replace(/^data:.*?base64,/,"");let i;if(typeof Buffer=="function"&&typeof Buffer.from=="function")i=Buffer.from(e,"base64");else if(typeof atob=="function"){const u=atob(e);i=new Uint8Array(u.length);for(let w=0;w<u.length;w++)i[w]=u.charCodeAt(w)}else throw new Error("Cannot decode base64-encoded data URL");o=await WebAssembly.instantiate(i,n)}else{const e=await fetch(t),i=e.headers.get("Content-Type")||"";if("instantiateStreaming"in WebAssembly&&i.startsWith("application/wasm"))o=await WebAssembly.instantiateStreaming(e,n);else{const u=await e.arrayBuffer();o=await WebAssembly.instantiate(u,n)}}return o.instance.exports};let l;function T(n){l=n}const s=new Array(128).fill(void 0);s.push(void 0,null,!0,!1);function r(n){return s[n]}let f=s.length;function S(n){n<132||(s[n]=f,f=n)}function y(n){const t=r(n);return S(n),t}const k=typeof TextDecoder>"u"?(0,p.require)("util").TextDecoder:TextDecoder;let h=new k("utf-8",{ignoreBOM:!0,fatal:!0});h.decode();let d=null;function L(){return(d===null||d.byteLength===0)&&(d=new Uint8Array(l.memory.buffer)),d}function g(n,t){return n=n>>>0,h.decode(L().subarray(n,n+t))}function c(n){f===s.length&&s.push(s.length+1);const t=f;return f=s[t],s[t]=n,t}function C(){l.testing_sin()}function a(n,t){try{return n.apply(this,t)}catch(o){l.__wbindgen_exn_store(c(o))}}function W(n,t){console.log(g(n,t))}function j(n){const t=r(n).crypto;return c(t)}function M(n){const t=r(n);return typeof t=="object"&&t!==null}function R(n){const t=r(n).process;return c(t)}function U(n){const t=r(n).versions;return c(t)}function q(n){y(n)}function B(n){const t=r(n).node;return c(t)}function F(n){return typeof r(n)=="string"}function O(){return a(function(){const n=p.require;return c(n)},arguments)}function V(n,t){const o=g(n,t);return c(o)}function E(n){const t=r(n).msCrypto;return c(t)}function P(){return a(function(n,t){r(n).randomFillSync(y(t))},arguments)}function D(){return a(function(n,t){r(n).getRandomValues(r(t))},arguments)}function N(n){return typeof r(n)=="function"}function z(n,t){const o=new Function(g(n,t));return c(o)}function H(){return a(function(n,t){const o=r(n).call(r(t));return c(o)},arguments)}function Z(){return a(function(){const n=self.self;return c(n)},arguments)}function $(){return a(function(){const n=window.window;return c(n)},arguments)}function I(){return a(function(){const n=globalThis.globalThis;return c(n)},arguments)}function K(){return a(function(){const n=global.global;return c(n)},arguments)}function Y(n){return r(n)===void 0}function G(){return a(function(n,t,o){const e=r(n).call(r(t),r(o));return c(e)},arguments)}function J(n){const t=r(n).buffer;return c(t)}function Q(n,t,o){const e=new Uint8Array(r(n),t>>>0,o>>>0);return c(e)}function X(n){const t=new Uint8Array(r(n));return c(t)}function nn(n,t,o){r(n).set(r(t),o>>>0)}function tn(n){const t=new Uint8Array(n>>>0);return c(t)}function en(n,t,o){const e=r(n).subarray(t>>>0,o>>>0);return c(e)}function rn(n){const t=r(n);return c(t)}function on(n,t){throw new Error(g(n,t))}function cn(){const n=l.memory;return c(n)}URL=globalThis.URL;const b=await A({"./micrograd_wasm_bg.js":{__wbg_log_dfdaa3ba4e25f49c:W,__wbg_crypto_1d1f22824a6a080c:j,__wbindgen_is_object:M,__wbg_process_4a72847cc503995b:R,__wbg_versions_f686565e586dd935:U,__wbindgen_object_drop_ref:q,__wbg_node_104a2ff8d6ea03a2:B,__wbindgen_is_string:F,__wbg_require_cca90b1a94a0255b:O,__wbindgen_string_new:V,__wbg_msCrypto_eb05e62b530a1508:E,__wbg_randomFillSync_5c9c955aa56b6049:P,__wbg_getRandomValues_3aa56aa6edec874c:D,__wbindgen_is_function:N,__wbg_newnoargs_e258087cd0daa0ea:z,__wbg_call_27c0f87801dedf93:H,__wbg_self_ce0dbfc45cf2f5be:Z,__wbg_window_c6fb939a7f436783:$,__wbg_globalThis_d1e6af4856ba331b:I,__wbg_global_207b558942527489:K,__wbindgen_is_undefined:Y,__wbg_call_b3ca7c6051f9bec1:G,__wbg_buffer_12d079cc21e14bdb:J,__wbg_newwithbyteoffsetandlength_aa4a17c33a06e5cb:Q,__wbg_new_63b92bc8671ed464:X,__wbg_set_a47bac70306a19a7:nn,__wbg_newwithlength_e9b4878cebadb3d3:tn,__wbg_subarray_a1f73cd4b5b42fe1:en,__wbindgen_object_clone_ref:rn,__wbindgen_throw:on,__wbindgen_memory:cn}},x),sn=b.memory,an=b.run,un=b.testing_sin,_n=b.__wbindgen_exn_store,v=b.__wbindgen_start,ln=Object.freeze(Object.defineProperty({__proto__:null,__wbindgen_exn_store:_n,__wbindgen_start:v,memory:sn,run:an,testing_sin:un},Symbol.toStringTag,{value:"Module"}));T(ln),v(),document.querySelector("#app").innerHTML=`
  <div>
    <a href="https://vitejs.dev" target="_blank">
      <img src="${_}" class="logo" alt="Vite logo" />
    </a>
    <a href="https://www.typescriptlang.org/" target="_blank">
      <img src="${m}" class="logo vanilla" alt="TypeScript logo" />
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
`;function fn(n){const t=()=>{C()};n.addEventListener("click",()=>t())}fn(document.querySelector("#counter"))})()});export default dn();