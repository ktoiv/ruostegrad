use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::math::computation::Computation;

use crate::nn::layer::Layer;
use crate::nn::layer::LayerConfig;

//Multi layered perceptron
pub struct MLP {
	layers: Vec<Layer>,
	computation: Computation,
	input_size: usize,
	output_size: usize,
	learning_rate: f32,
}


impl MLP {

	pub fn new(input_size: usize, output_size: usize, layer_configs: Vec<LayerConfig>, learning_rate: f32) -> Self {
		let mut computation = Computation {
			tensors: Vec::new(),
			ops: Vec::new(),
		};

		// First n tensors will be reserved for the input where n is the size of the
		let input_indices: Vec<usize> =(0..input_size)
			.map(|_| computation.add_tensor(0.0))
			.collect();
	
		let mut layers = Vec::new();

		let first_layer_config = &layer_configs[0];
		let first_layer = Layer::new(input_indices, first_layer_config.output_size, first_layer_config.activation_function, &mut computation);

		layers.push(first_layer);

		for i in 1..layer_configs.len() {
			let previous_layer_output_indices: Vec<usize> = layers[i - 1].get_output_indices();
			let layer_config: &LayerConfig = &layer_configs[i];

			let layer = Layer::new(previous_layer_output_indices, layer_config.output_size, layer_config.activation_function, &mut computation);
			layers.push(layer);
		}

		MLP{
			layers,
			computation,
			input_size,
			output_size,
			learning_rate
		}
	}

	fn get_output(&self) -> Vec<f32> {
		let output_indices = self.layers.last().unwrap().get_output_indices();

		output_indices
			.iter()
			.map(|&index| self.computation.get_tensor_value(index))
			.collect()
	}

	fn mean_squared_error(&self, actual: &Vec<f32>, predicted: &Vec<f32>) -> f32 {
		if actual.len() != predicted.len() {
			panic!("Predicted data size does not match to the actual data");
		}

		let sum_of_squares = actual.iter()
			.zip(predicted.iter())
			.map(|(a,p)| (a-p).powi(2) )
			.sum::<f32>();

		let mse = sum_of_squares / (actual.len() as f32);

		mse
	}


	fn update_weights(&mut self) {

		for layer in &mut self.layers {
			for neuron in &layer.neurons {
				for &weight_index in &neuron.weight_indices {
					let weigth_tensor = &mut self.computation.tensors[weight_index];
					weigth_tensor.data -= self.learning_rate * weigth_tensor.grad;
				}

				let bias_tensor = &mut self.computation.tensors[neuron.bias_index];
				bias_tensor.data -= self.learning_rate * bias_tensor.grad;
			}
		}

	}

	pub fn train(&mut self, input_data: Vec<Vec<f32>>, targets: Vec<Vec<f32>>, epochs: i32) -> Result<(), &'static str> {

		if input_data.len() != targets.len() {
			return Err("Input data and targets have to be the same length");
		}

		for input in &input_data {
			if input.len() != self.input_size {
				return Err("Input data size has to match the neural network input size");
			}
		}

		for target in &targets {
			if target.len() != self.output_size {
				return Err("Target size has to match the neural network output size");
			}
		}
		
		let mut rng = thread_rng();

		for epoch in 0..epochs {

			let mut combined_data: Vec<(Vec<f32>, Vec<f32>)> = input_data.iter().cloned().zip(targets.iter().cloned()).collect();
			combined_data.shuffle(&mut rng);

			let (shuffled_inputs, shuffled_targets): (Vec<_>, Vec<_>) = combined_data.into_iter().unzip();

			let mut total_loss: f32 = 0.0;

			for (input, target) in shuffled_inputs.iter().zip(shuffled_targets.iter()) {

				for (i, &value) in input.iter().enumerate() {
					self.computation.set_tensor_value(i, value);
				}

				self.computation.forward();

				let predicted: Vec<f32> = self.get_output();
				let loss = self.mean_squared_error(&target, &predicted);
				total_loss += loss;

				let loss_grad: Vec<f32> = predicted.iter()
					.zip(target.iter())
					.map(|(p, t)| 2.0 * (p - t))
					.collect();

				//println!("{:?}", loss_grad);

				let output_indices = self.layers.last().unwrap().get_output_indices();

				for (&index, &grad) in output_indices.iter().zip(&loss_grad) {
					self.computation.tensors[index].grad = grad;
				}
	
				self.computation.backward();

				self.update_weights();

				self.computation.zero_grad();
			}

			let avg_loss: f32 = total_loss / (input_data.len() as f32);
			println!("Epoch {}: Average loss = {}", epoch, avg_loss);
		}

        Ok(())
	}

	pub fn predict(&mut self, input: Vec<f32>) -> Vec<f32> {
		for (i, &value) in input.iter().enumerate() {
			self.computation.set_tensor_value(i, value);
		}

		self.computation.forward();

		let predicted: Vec<f32> = self.get_output().clone();

		predicted
	}

}