// value ref(value inner)
// value can be cloned uses rc to clone address on stack not value on heap
// value inner

extern crate rand;
use std::{
  cell::{Ref, RefCell},
  collections::HashSet,
  fmt,
  ops::{Add, BitXor, Mul, Neg, Sub},
  rc::Rc,
}; // 0.8.5

type BackwardFn = fn(value: &Ref<ValueInner>);

struct ValueInner {
  pub id: u64,
  pub data: f32,
  pub previous: Vec<Value>,
  pub label: String,
  pub gradient: f32,
  // closure
  pub _backward: Option<BackwardFn>,
}

impl ValueInner {
  fn new(
    data: f32,
    previous: Vec<Value>,
    label: String,
    _backward: Option<BackwardFn>,
  ) -> ValueInner {
    let id = rand::random();
    let gradient = 0.0;
    ValueInner {
      data,
      previous,
      label,
      gradient,
      _backward,
      id,
    }
  }
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueInner>>);

impl fmt::Display for Value {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let data = self.0.borrow().data;
    let label = &self.0.borrow().label;
    let gradient = self.0.borrow().gradient;
    let id = self.0.borrow().id;
    write!(
      f,
      "Value( label={}, data={} , grad={}, id={} )",
      label, data, gradient, id
    )
  }
}

impl Value {
  // resets grad for self and all children
  pub fn zero_grad(&self) {
    self.0.borrow_mut().gradient = 0.0;
    self.0.borrow().previous.iter().for_each(|child| {
      child.zero_grad();
    });
  }

  pub fn draw(&self) {
    println!("{}", self);
    self.0.borrow().previous.iter().for_each(|child| {
      child.draw();
    });
  }

  pub fn set_gradient(&self, gradient: f32) {
    self.0.borrow_mut().gradient = gradient;
  }

  pub fn from(data: f32) -> Value {
    Value(Rc::new(RefCell::new(ValueInner::new(
      data,
      vec![],
      "".to_string(),
      None,
    ))))
  }

  fn build_topo(&self, topo: &mut Vec<Value>, visited: &mut HashSet<u64>) {
    let id = self.0.borrow().id;
    if !visited.contains(&id) {
      visited.insert(id);
      for child in &self.0.borrow().previous {
        child.build_topo(topo, visited);
      }
      topo.push(self.clone());
    }
  }

  pub fn backward(&self) {
    let mut topo: Vec<Value> = Vec::new();

    let mut visited: HashSet<u64> = std::collections::HashSet::new();

    self.build_topo(&mut topo, &mut visited);

    self.0.borrow_mut().gradient = 1.0;
    for value in topo.iter().rev() {
      let borrowed_value = value.0.borrow();
      if let Some(backward_fn) = borrowed_value._backward {
        backward_fn(&borrowed_value);
      }
    }
  }

  pub fn set_label(self, label: &str) -> Value {
    self.0.borrow_mut().label = label.to_string();
    self
  }

  pub fn data(&self) -> f32 {
    self.0.borrow().data
  }

  pub fn gradient(&self) -> f32 {
    self.0.borrow().gradient
  }

  pub fn adjust(&self, delta: f32) {
    let mut value = self.0.borrow_mut();
    value.data += delta;
  }

  pub fn tanh(&self) -> Value {
    let data = self.0.borrow().data.tanh();
    let label = format!("tanh({})", self.0.borrow().label);
    let previous = vec![self.clone()];
    let _backward: Option<BackwardFn> = Some(|value| {
      let mut previous = value.previous[0].0.borrow_mut();
      previous.gradient += (1.0 - value.data.powf(2.0)) * value.gradient;
    });

    let out = Value(Rc::new(RefCell::new(ValueInner::new(
      data, previous, label, _backward,
    ))));

    out
  }
}

impl Add<Value> for Value {
  type Output = Value;

  fn add(self, other: Value) -> Self::Output {
    add(&self, &other)
  }
}

// question: why do we need to implement 'b here
// it should be clear that add creates a new value with independent data
impl<'a, 'b> Add<&'b Value> for &'a Value {
  type Output = Value;

  fn add(self, other: &'b Value) -> Self::Output {
    add(&self, other)
  }
}

impl Sub<Value> for Value {
  type Output = Value;

  fn sub(self, other: Value) -> Self::Output {
    add(&self, &(-other))
  }
}

impl<'b> Sub<&'b Value> for &Value {
  type Output = Value;

  fn sub(self, other: &'b Value) -> Self::Output {
    add(self, &(-other))
  }
}

fn add(a: &Value, b: &Value) -> Value {
  let data = a.0.borrow().data + b.0.borrow().data;
  let label = format!("({}+{})", a.0.borrow().label, b.0.borrow().label);
  let previous = vec![a.clone(), b.clone()];
  let _backward: Option<BackwardFn> = Some(|value| {
    let mut first = value.previous[0].0.borrow_mut(); // a
    let mut second = value.previous[1].0.borrow_mut(); // b
    first.gradient += value.gradient;
    second.gradient += value.gradient;
  });

  let out = Value(Rc::new(RefCell::new(ValueInner::new(
    data, previous, label, _backward,
  ))));

  out
}

impl Mul<Value> for Value {
  type Output = Value;

  fn mul(self, other: Value) -> Self::Output {
    mul(&self, &other)
  }
}

impl<'b> Mul<&'b Value> for &Value {
  type Output = Value;

  fn mul(self, other: &'b Value) -> Self::Output {
    mul(self, other)
  }
}

impl Neg for Value {
  type Output = Value;

  fn neg(self) -> Self::Output {
    mul(&self, &Value::from(-1.0))
  }
}

impl<'b> Neg for &'b Value {
  type Output = Value;

  fn neg(self) -> Self::Output {
    mul(self, &Value::from(-1.0))
  }
}

fn mul(a: &Value, b: &Value) -> Value {
  let data = a.0.borrow().data * b.0.borrow().data;
  let label = format!("({}*{})", a.0.borrow().label, b.0.borrow().label);
  let previous = vec![a.clone(), b.clone()];
  let _backward: Option<BackwardFn> = Some(|value| {
    let mut first = value.previous[0].0.borrow_mut(); // a
    let mut second = value.previous[1].0.borrow_mut(); // b

    first.gradient += second.data * value.gradient;
    second.gradient += first.data * value.gradient;
  });

  let out = Value(Rc::new(RefCell::new(ValueInner::new(
    data, previous, label, _backward,
  ))));

  out
}

impl BitXor<Value> for Value {
  type Output = Value;

  fn bitxor(self, other: Value) -> Self::Output {
    pow(&self, &other)
  }
}

impl<'b> BitXor<&'b Value> for &Value {
  type Output = Value;

  fn bitxor(self, other: &'b Value) -> Self::Output {
    pow(self, other)
  }
}

fn pow(base: &Value, exponent: &Value) -> Value {
  let data = base.0.borrow().data.powf(exponent.0.borrow().data);
  let label = format!("({}^{})", base.0.borrow().label, exponent.0.borrow().label);
  let previous = vec![base.clone(), exponent.clone()];
  // y = x^a
  // dy/dx = a * x^(a-1)
  let _backward: Option<BackwardFn> = Some(|value| {
    let mut base = value.previous[0].0.borrow_mut();
    let exponent = value.previous[1].0.borrow();

    base.gradient += exponent.data * base.data.powf(exponent.data - 1.0) * value.gradient;
  });

  let out = Value(Rc::new(RefCell::new(ValueInner::new(
    data, previous, label, _backward,
  ))));

  out
}
