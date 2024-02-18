use crate::math::operation::Operation;
use crate::math::tensor::Tensor;

pub struct Computation {
    pub tensors: Vec<Tensor>,
    pub ops: Vec<Operation>,
}

impl Computation {
    pub fn add_operation(&mut self, op: Operation) {
        self.ops.push(op);
    }

    //Add a new tensor to the computation graph and return the index of it
    pub fn add_tensor(&mut self, initial_value: f32) -> usize {
        let index: usize = self.tensors.len();
        self.tensors.push(Tensor {
            data: initial_value,
            grad: 0.0,
        });

		return index;
    }

	pub fn forward(&mut self) {
    	for op in &self.ops {
        	let output_data: f32 = op.forward(self);
			match op {
                Operation::Add { output, .. }
                | Operation::Mul { output, .. }
                | Operation::Tanh { output, .. }
                | Operation::Sigmoid { output, .. }
                | Operation::ReLU { output, .. } => self.tensors[*output].data = output_data,
            }
    	}
	}

	pub fn backward(&mut self) {
		for op in self.ops.iter().rev() {

			let grad_changes = op.backward(self);

			for (index, grad_change) in grad_changes {
				self.tensors[index].grad += grad_change;
			}
		}
	}

	pub fn get_tensor_value(&self, index: usize) -> f32 {
		self.tensors[index].data
	}

	pub fn set_tensor_value(&mut self, index: usize, value: f32) {
		if let Some(tensor) = self.tensors.get_mut(index) {
			tensor.data = value;
		} else {
			panic!("Trying to assign value to a non existing tensor");
		}
	}


	pub fn zero_grad(&mut self) {
		for tensor in &mut self.tensors {
			tensor.grad = 0.0;
		}
	}
}