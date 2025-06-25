
class LinearLayerModel:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = [[0.0] * output_size for _ in range(input_size)]
        self.biases = [0.0] * output_size

    def forward(self, inputs):
        outputs = [0.0] * self.output_size
        for i in range(self.output_size):
            for j in range(self.input_size):
                outputs[i] += inputs[j] * self.weights[j][i]
            outputs[i] += self.biases[i]
        return outputs
    
weights = [
    [0.2, 0.8, -0.5],
    [0.5, -0.91, 0.26],
    [-0.26, -0.27, 0.17]
]
biases = [0.1, -0.2, 0.3]
model = LinearLayerModel(3, 3)
model.weights = weights
model.biases = biases
inputs = [1.0, 2.0, 3.0]
outputs = model.forward(inputs)
print("Outputs:", outputs)