use crate::value_next::Value;
use rand::{thread_rng, Rng};

#[derive(Clone)]
pub struct Neuron {
  weights: Vec<Value>,
  bias: Value,
  with_activation: bool,
}

impl Neuron {
  pub fn new(inputs: i32, with_activation: bool) -> Neuron {
    let mut weights: Vec<Value> = Vec::new();
    let bias = Value::from(thread_rng().gen_range(-1.0..1.0)).set_label("b");
    for i in 0..inputs {
      // -1.0 to 1.0
      let w = thread_rng().gen_range(-1.0..1.0);
      weights.push(Value::from(w).set_label(&format!("w{}", i)));
    }
    Neuron { weights, bias, with_activation }
  }

  pub fn forward(&self, inputs: &Vec<Value>) -> Value {
    // panic if inputs.len() != self.weights.len()
    if inputs.len() != self.weights.len() {
      panic!("inputs.len() != self.weights.len()");
    }
    let mut sum = self.bias.clone();
    for i in 0..self.weights.len() {
      sum = sum + self.weights[i].clone() * inputs[i].clone();
    }

    if self.with_activation {
      return sum.tanh();
    } else {
      return sum;
    }
  }

  pub fn parameters(&self) -> Vec<Value> {
    let mut params: Vec<Value> = Vec::new();
    params.push(self.bias.clone());
    for w in self.weights.iter() {
      params.push(w.clone());
    }
    params
  }

  pub fn draw(&self) {
    println!("Neuron len: {}", self.weights.len());
    for w in self.weights.iter() {
      println!("{}", w);
    }
    println!("{}", self.bias);
  }
}

pub struct Layer {
  neurons: Vec<Neuron>,
  label: String,
  inputs: i32,
  outputs: i32,
}

impl Layer {
  // create a new layer with inputs and outputs
  // inputs are the number of weights per neuron
  // outputs are the number of neurons
  pub fn new(inputs: i32, outputs: i32, label: String, with_activation: bool) -> Layer {
    let mut neurons: Vec<Neuron> = Vec::new();
    for _ in 0..outputs {
      neurons.push(Neuron::new(inputs, with_activation));
    }

    Layer { neurons, label, inputs, outputs }
  }

  pub fn parameters(&self) -> Vec<Value> {
    let mut params: Vec<Value> = Vec::new();
    for n in self.neurons.iter() {
      let mut n_params = n.parameters();
      params.append(&mut n_params);
    }
    params
  }

  pub fn forward(&self, inputs: &Vec<Value>) -> Vec<Value> {
    let mut outputs: Vec<Value> = Vec::new();

    for n in self.neurons.iter() {
      outputs.push(n.forward(inputs));
    }
    outputs
  }

  pub fn draw(&self) {
    println!("Layer {} inputs: {}, outputs: {}", self.label, self.inputs, self.outputs);
  }
}

pub struct MLP {
  layers: Vec<Layer>, // last layer is the output layer
}

impl MLP {
  pub fn new(inputs: i32, layers: Vec<i32>) -> MLP {
    let mut _layers: Vec<Layer> = Vec::new();
    let mut i = 0; let max = layers.len();
    for layer_height in layers {
      let label = format!("Layer{}", i);
      let layer_inputs = if i == 0 { inputs } else { _layers[i - 1].outputs };
      let with_activation = i != max - 1;
      _layers.push(Layer::new(layer_inputs, layer_height, label, with_activation));
      i += 1;
    }

    MLP { layers: _layers }
  }

  pub fn draw(&self) {
    println!("MLP");
    for l in self.layers.iter() {
      l.draw();
    }
  }

  pub fn parameters(&self) -> Vec<Value> {
    let mut params: Vec<Value> = Vec::new();
    for l in self.layers.iter() {
      let mut l_params = l.parameters();
      params.append(&mut l_params);
    }
    params
  }

  pub fn forward(&self, inputs: &Vec<Value>) -> Vec<Value> {
    let mut outputs = inputs.clone();
    for l in self.layers.iter() {
      outputs = l.forward(&outputs);
    }
    outputs
  }

  // private static method
  // calc loss of single example
  fn loss(ytruth: &Vec<Value>, ypreds: &Vec<Value>) -> Value {
    let mut loss = Value::from(0.0).set_label("loss");
    for i in 0..ytruth.len() {
      let y = ytruth[i].clone();
      let ypred = ypreds[i].clone();
      loss = (loss + (y - ypred))^Value::from(2.0).set_label("2");
    }
    loss
  }

  pub fn updateWeights(&self, step_size: f32) {
    self.parameters().iter().for_each(|p| {
      p.adjust(-step_size * p.gradient());
    });
  }

  // calc total loss of all examples
  pub fn total_loss(&self, xs: &Vec<Vec<Value>>, ys: &Vec<Vec<Value>>) -> Value {
    let total_loss = xs.iter().zip(ys.iter()).map(|(x,y)| {
      let y_pred = self.forward(x);
      let loss: Value = MLP::loss(y, &y_pred);
      loss
    }).reduce(|acc, x| acc + x).unwrap().set_label("total_loss");

    total_loss * Value::from(1.0 / xs.len() as f32).set_label("1/n")
  }

  // single traning loop for a given loss value
  // zero grad, backprop, update
  pub fn traning_loop(&self, total_loss: &Value) {
    total_loss.zero_grad();
    total_loss.set_gradient(1.0);
    total_loss.backward();
    self.updateWeights(0.01);
  }

}