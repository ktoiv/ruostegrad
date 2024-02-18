use crate::nn::{layer::LayerConfig, mlp::MLP, activation_function::ActivationFunction};

pub mod math;
pub mod nn;


fn main() -> Result<(), &'static str> {
    // Define the input size, output size, and learning rate
    let input_size = 3;
    let output_size = 1;
    let learning_rate = 0.1;
    let epochs = 100;

    // Define layer configurations
    let layer_configs = vec![
        LayerConfig { output_size: 4, activation_function: ActivationFunction::Tanh },
        LayerConfig { output_size: 4, activation_function: ActivationFunction::Tanh },
        LayerConfig { output_size: output_size, activation_function: ActivationFunction::Tanh },
    ];

    // Create an instance of the MLP
    let mut mlp = MLP::new(input_size, output_size, layer_configs, learning_rate);

    // Example training data
    let input_data = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
        // Add more data points as needed
    ];
    let targets = vec![
        vec![-1.0],
        vec![1.0],
        vec![1.0],
        vec![-1.0],
        // Add more targets as needed
    ];

    // Train the MLP
    match mlp.train(input_data, targets, epochs) {
        Ok(_) => println!("Training was successful!"),
        Err(e) => eprintln!("An error occurred during training: {}", e),
    }

    // Example prediction
    let input = vec![0.5, 0.5, 0.5];
    let prediction = mlp.predict(input);
    println!("Prediction: {:?}", prediction);

    Ok(())
}
