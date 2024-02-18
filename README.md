# Ruostegrad

This is a simple implementation of a neural net in Rust.

It implements a multi layered perceptron (MLP) that can be trained using gradient descent and back propagation. 

## Running the project

The project comes with a simple excample case defined in `src/main.rs`. it creates a very small and simple MLP and uses it to predict a value.

To run it from project root:

 type first `cargo build`

then `cargo run`


## Implementation


### Tensor
The most simple unit in the neural network. It consists of a value and a gradient. Gradient describes the impact a particular tensor has on the overall prediction in the neural net

### Operation
Describes a single mathematical operation performed on a tensor. An operation can be of the following types:

- Add
- Multiply
- Tanh
- Sigmoid
- ReLu

Each operation have one or multiple input tensors and one output tensor. All operations have two functions `forward` and `backward`.

A forward pass performs the actual mathematical operation on the input tensors values and stores the result into the output tensor's value

A backward pass calculates the derivative of the mathematical operations using the input values and stores the result multiplied by the output gradient into the input tensors gradients.

### Computation

Keeps track of all the tensors and operations in the neural net. Has a list of tensors and operations. Tensors are also passed to an operation by providing it with the indices of the tensors in the computation and the computation itself. This way the whole chain of computation can be traversed in both direction in a simple manner.


### Neuron

Consists of input values, input weights, output tensors, an activation function and a bias.

#### Forward pass
 Each input value has to have a corresponding weight value. Each input value is multiplied with its weight value and a separate multiplication operation is created for each. Then the results of those are added together in separate operation. The bias is added to the result of this operation as another add operation. Then finally the activation function which is also an operation is applied to the result of the previous operation

#### Backward pass
The previous operations are traversed in inverse order and the `backward` function for each operation is called in succession


### LayerConfig
A helper struct that holds information about a layer's output size and which activation function is applied to all of the neurons in a layer.

### Layer
Has a list of input neurons and a LayerConfig. All the neurons in a layer share the activation function with each other. 


### MLP
The actual net. Takes in the size of the input layer and a list of layer configs. The output size is determined by the output size of the last layer. 

#### Creating a MLP

Here we create a MLP that takes in three numbers and tries to predict a single value from them
```rust
// Define the input size, output size, and learning rate
let input_size = 3;
let output_size = 1;
let learning_rate = 0.1;
// Define layer configurations
let layer_configs = vec![
    LayerConfig { output_size: 4, activation_function: ActivationFunction::Tanh },
    LayerConfig { output_size: 4, activation_function: ActivationFunction::Tanh },
    LayerConfig { output_size: output_size, activation_function: ActivationFunction::Tanh },
];
// Create an instance of the MLP
let mut mlp = MLP::new(input_size, output_size, layer_configs, learning_rate);
```


#### Training
To train the MLP provide it with training data. Training data consists of data points as input and a corresponding target which the net tries to learn how to predict. The input data consists of vectors which must be the same length as the input size defined when creating the MLP.

```rust
let input_data = vec![
    vec![2.0, 3.0, -1.0],
    vec![3.0, -1.0, 0.5],
    vec![0.5, 1.0, 1.0],
    vec![1.0, 1.0, -1.0],
];
let targets = vec![
    vec![-1.0],
    vec![1.0],
    vec![1.0],
    vec![-1.0],
];
let epochs = 100;

match mlp.train(input_data, targets, epochs) {
    Ok(_) => println!("Training was successful!"),
    Err(e) => eprintln!("An error occurred during training: {}", e),
}
```

#### Predicting

To predict values with a trained neural net, call its predict function with a vector of input values which size needs to match the input size defined when creating a MLP
```rust
let input = vec![0.5, 0.5, 0.5];
let prediction = mlp.predict(input);
println!("Prediction: {:?}", prediction);
```