use rand::random;

use crate::math::computation::Computation;

use crate::nn::neuron::Neuron;
use crate::nn::activation_function::ActivationFunction;

pub struct Layer {
	pub neurons: Vec<Neuron>,
}

impl Layer {

	pub fn new(input_indices: Vec<usize>, output_size: usize, activation_function: ActivationFunction, computation: &mut Computation) -> Self {
		let mut neurons = Vec::new();

		// The output size determines the number of neurons in this layer
		for _ in 0..output_size {

			let mut weight_indices = Vec::new();

			// This is a dense layer so every input goes to every neuron 
			for _ in &input_indices {
				let weight = random::<f32>() * 2.0 - 1.0;
				weight_indices.push(computation.add_tensor(weight));
			}
			
			
			let neuron = Neuron::new(input_indices.clone(), weight_indices, activation_function, computation);
			neurons.push(neuron);

		}

		Layer {
			neurons
		}
	}

	pub fn get_output_indices(&self) -> Vec<usize> {
		self.neurons.iter().map(|neuron| neuron.output_index).collect()
	}

}


pub struct LayerConfig {
	pub activation_function: ActivationFunction,
	pub output_size: usize,
}