mod nn;
mod value;

use nn::{MLP};
use value::Value;

fn testing_sin() {
  // data sin(x) from -10 to 100
  let mut xs = Vec::new();
  let mut ys = Vec::new();
  for i in -10..10 {
    let x = Value::from(i as f32).set_label("x");
    xs.push(vec![x]);
    let y = Value::from((i as f32).sin()).set_label("y");
    ys.push(vec![y]);
  }

  let mlp = MLP::new(1, vec![3, 3, 1]);
  println!("Model is created!");
  mlp.draw();
  println!("Training...");

  for i in 0..50 {
    let loss = mlp.total_loss(&xs, &ys);
    mlp.training_loop(&loss);
    println!("Loss {}: {}", i, loss.data());
  }

  println!("Model is trained!");
  println!("Let's test it!");

  println!("sin(0) = {}; mlp(0) = {}", 0.0_f32.sin(), mlp.forward(&vec![Value::from(0.0).set_label("x")])[0].data());
  println!("sin(-0.1) = {}; mlp(-0.1) = {}", (-0.1_f32).sin(), mlp.forward(&vec![Value::from(-0.1).set_label("x")])[0].data());
  println!("sin(0.1) = {}; mlp(0.1) = {}", 0.1_f32.sin(), mlp.forward(&vec![Value::from(0.1).set_label("x")])[0].data());
  println!("sin(0.5) = {}; mlp(0.5) = {}", 0.5_f32.sin(), mlp.forward(&vec![Value::from(0.5).set_label("x")])[0].data());
  println!("sin(1) = {}; mlp(1) = {}", 1.0_f32.sin(), mlp.forward(&vec![Value::from(1.0).set_label("x")])[0].data());
  println!("sin(3.14) = {}; mlp(3.14) = {}", 3.14_f32.sin(), mlp.forward(&vec![Value::from(3.14).set_label("x")])[0].data());
  println!("sin(-3.14/2) = {}; mlp(-3.14/2) = {}", (-3.14_f32/2.0_f32).sin(), mlp.forward(&vec![Value::from(-3.14/2.0).set_label("x")])[0].data());

}

fn main() {
  testing_sin();
}