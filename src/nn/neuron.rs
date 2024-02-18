use rand::random;
use crate::math::operation::Operation;
use crate::math::computation::Computation;
use crate::nn::activation_function::ActivationFunction;

pub struct Neuron {
    pub input_indices: Vec<usize>,
    pub weight_indices: Vec<usize>,
	pub bias_index: usize,
    pub output_index: usize,
}


impl Neuron {

	pub fn new(input_indices: Vec<usize>, weight_indices: Vec<usize>, activation_function: ActivationFunction, computation: &mut Computation) -> Self {
		let bias: f32 = random::<f32>() * 2.0 - 1.0;
		let bias_index: usize = computation.add_tensor(bias);

		/* The output of the neuron will be the result of the activation function 
		   when it is applied to the sum of all inputs * weights + bias */
		let output_index: usize = computation.add_tensor(0.0);

		let mut mul_outputs: Vec<usize> = Vec::new();

		for i in 0..input_indices.len() {
			let mul_output: usize = computation.add_tensor(0.0);

			let mul_op = Operation::Mul {
				inputs: vec![input_indices[i], weight_indices[i]],
				output: mul_output,
			};

			computation.add_operation(mul_op);
			mul_outputs.push(mul_output);
		}

		let mut previous_output = output_index;

		for &mul_output in &mul_outputs {
			let add_output: usize = computation.add_tensor(0.0);
			let add_op = Operation::Add { inputs: vec![previous_output, mul_output], output: add_output };

			computation.add_operation(add_op);
			previous_output = add_output;
		}


		let bias_output: usize = computation.add_tensor(0.0);
		let bias_op = Operation::Add {
			inputs: vec![bias_index, previous_output],
			output: bias_output
		};

		computation.add_operation(bias_op);

		let activated_output_index: usize = computation.add_tensor(0.0);
		
		let activation_op = match activation_function {
			ActivationFunction::Sigmoid => Operation::Sigmoid { input: bias_output, output: activated_output_index },
			ActivationFunction::Tanh => Operation::Tanh { input: bias_output, output: activated_output_index },
			ActivationFunction::ReLU => Operation::ReLU { input: bias_output, output: activated_output_index },
		};

		computation.add_operation(activation_op);

		Neuron {
			input_indices,
			weight_indices,
			bias_index,
			output_index: activated_output_index,
		}
	}
}