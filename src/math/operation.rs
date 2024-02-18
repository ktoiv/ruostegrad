use std::collections::HashMap;

use crate::math::computation::Computation;

#[derive(Clone)]
pub enum Operation {
    Add { inputs: Vec<usize>, output: usize },
    Mul { inputs: Vec<usize>, output: usize },
    Tanh { input: usize, output: usize },
    Sigmoid { input: usize, output: usize },
    ReLU { input: usize, output: usize }
}

impl Operation {
	pub fn forward(&self, computation: &Computation) -> f32 {
		match self {
			Operation::Add { inputs, output: _ } => {
				inputs
					.iter()
					.map(|&i| computation.tensors[i].data)
					.sum()

			},
			Operation::Mul { inputs, output: _ } => {
				assert!(inputs.len() == 2);

				inputs
					.iter()
					.map(|&i| computation.tensors[i].data)
					.product()

			},
			Operation::Tanh { input, output: _ } => {
				let value: f32 = computation.tensors[*input].data;
				value.tanh()
			},
			Operation::Sigmoid { input, output: _ } => {
				let value: f32 = computation.tensors[*input].data;
				1.0 / (1.0 / (-value).exp())
			},
			Operation::ReLU { input, output: _ } => {
				let value = computation.tensors[*input].data;
				value.max(0.0)
			}
		}
	}

	pub fn backward(&self, computation: &Computation) -> HashMap<usize, f32> {
		
		let mut grad_changes: HashMap<usize, f32> = HashMap::new();
		
		match self {
			Operation::Add { inputs, output } => {
				assert!(inputs.len() == 2);
	
				let output_grad = computation.tensors[*output].grad;
				let (first_input, second_input) = (inputs[0], inputs[1]);
				/* The derivative of each input in relation to the output is always 1
				 in addition so we can omit the multiplication by one and simply add the 
				 gradient from the output */

				 grad_changes.insert(first_input, output_grad);
				 grad_changes.insert(second_input, output_grad);
			},
			Operation::Mul { inputs, output } => {
				assert!(inputs.len() == 2);

				let output_grad = computation.tensors[*output].grad;
				let (first_input, second_input) = (inputs[0], inputs[1]);

				let first_value = computation.tensors[first_input].data;
				let second_value = computation.tensors[second_input].data;

				/* In multiplication the local gradient is equal to the value of the other input.
				 Because we want to backpropagate here we will multiply the other input by the
				 grad of the result tensor */

				 grad_changes.insert(first_input, second_value * output_grad);
				 grad_changes.insert(second_input, first_value * output_grad);
			},
			Operation::Tanh { input, output } => {
				let output_grad: f32 = computation.tensors[*output].grad;
				let input_value: f32 = computation.tensors[*input].data;

				let cosh = input_value.cosh();
				let seq_squared = 1.0 / cosh.powi(2);

				grad_changes.insert(*input, seq_squared * output_grad);
			}
			/*
				https://theneuralblog.com/derivative-sigmoid-function
				"Therefore, the derivative of a sigmoid function is equal
				 to the multiplication of the sigmoid function itself
				 with (1 – sigmoid function itself)"
			*/
			Operation::Sigmoid { input, output } => {
				let output_grad: f32 = computation.tensors[*output].grad;
				let input_value: f32 = computation.tensors[*input].data;

				let sigmoid = 1.0 / (1.0 / (-input_value).exp());
				let sigmoid_derivative = sigmoid * (1.0 - sigmoid); 

				grad_changes.insert(*input, sigmoid_derivative * output_grad);

			},
			Operation::ReLU { input, output } => {
				let output_grad: f32 = computation.tensors[*output].grad;
				let input_value: f32 = computation.tensors[*input].data;
				let relu_derivative: f32 = if input_value > 0.0 { 1.0 } else { 0.0 };
				grad_changes.insert(*input, relu_derivative * output_grad);
			}
		}

		grad_changes
	}
}