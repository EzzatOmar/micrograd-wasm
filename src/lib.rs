mod utils;
mod nn;
mod value;

use nn::{MLP};
use value::Value;

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
fn run() {
  // run stuff on load here
}

// https://rustwasm.github.io/wasm-bindgen/examples/console-log.html
#[wasm_bindgen]
extern "C" {
  fn alert(s: &str);

  // Use `js_namespace` here to bind `console.log(..)` instead of just
  // `log(..)`
  #[wasm_bindgen(js_namespace = console)]
  fn log(s: &str);

  // The `console.log` is quite polymorphic, so we can bind it with multiple
  // signatures. Note that we need to use `js_name` to ensure we always call
  // `log` in JS.
  #[wasm_bindgen(js_namespace = console, js_name = log)]
  fn log_u32(a: u32);

  // Multiple arguments too!
  #[wasm_bindgen(js_namespace = console, js_name = log)]
  fn log_many(a: &str, b: &str);
}


// Next let's define a macro that's like `println!`, only it works for
// `console.log`. Note that `println!` doesn't actually work on the wasm target
// because the standard library currently just eats all output. To get
// `println!`-like behavior in your app you'll likely want a macro like this.

macro_rules! console_log {
  // Note that this is using the `log` function imported above during
  // `bare_bones`
  ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
pub fn greet() {
  alert("Hello, micrograd-wasm!");
}

#[wasm_bindgen]
pub fn testing_sin() {
  let mlp = MLP::new(1, vec![3, 3, 1]);
  console_log!("Model is created! 1 x 3 x 3 x 1");
  
  let mut xs = Vec::new();
  let mut ys = Vec::new();
  for i in -10..10 {
    let x = Value::from(i as f32).set_label("x");
    xs.push(vec![x]);
    let y = Value::from((i as f32).sin()).set_label("y");
    ys.push(vec![y]);
  }
  console_log!("Setup data for sin {}", &xs[0][0].data());

  
  console_log!("Training...");
  for i in 0..10000 {
    let loss = mlp.total_loss(&xs, &ys);
    mlp.training_loop(&loss);
    console_log!("Loss {}: {}", i, loss.data());
  }

  console_log!("Model is trained!");
  console_log!("Let's test it!");

  for i in [0.0, -0.1, 0.1, 0.5, 1.0, 3.14, -3.14/2.0].iter() {
    let x:f32 = *i;
    let y_truth = x.sin();
    let y_pred = mlp.forward(&vec![Value::from(x).set_label("x")])[0].data();
    let error = (y_truth - y_pred).abs();
    console_log!("sin({}) = {}; mlp({}) = {}; error = {}", x, y_truth, x, y_pred, error);
  }
}
