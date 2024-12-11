import math
import random
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
import csv

class DataPreprocessor:
    def __init__(self):
        self.data = []
        self.features = []
        self.targets = []
        self.training_data = []
        self.testing_data = []

    def load_csv(self, file_path):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            self.data = list(reader)

    def preprocess(self):
        if not self.data:
            raise ValueError("No data loaded.")

        # Assume first row is header, last column is target
        self.features = [[float(x) for x in row[:-1]] for row in self.data[1:]]
        self.targets = [[int(row[-1])] for row in self.data[1:]]

        # Normalize features between 0 and 1
        for i in range(len(self.features[0])):
            column = [row[i] for row in self.features]
            min_val, max_val = min(column), max(column)
            for row in self.features:
                if max_val != min_val:
                    row[i] = (row[i] - min_val) / (max_val - min_val)
                else:
                    row[i] = 0.0

    def split_data(self, train_ratio=0.8):
        data = list(zip(self.features, self.targets))
        random.shuffle(data)
        split_index = int(len(data) * train_ratio)
        self.training_data = data[:split_index]
        self.testing_data = data[split_index:]


class Neuron:
    def __init__(self, layer_index, neuron_index, x, y):
        self.layer_index = layer_index
        self.neuron_index = neuron_index
        self.x = x
        self.y = y
        self.output = 0.0
        self.delta = 0.0
        self.bias = random.uniform(-1, 1)

    def draw(self, canvas):
        canvas.create_oval(self.x - 15, self.y - 15, self.x + 15, self.y + 15, fill='white', outline='black')
        canvas.create_text(self.x, self.y, text=f"{self.output:.2f}", font=("Arial", 8))


class Weight:
    def __init__(self, from_neuron, to_neuron):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.value = random.uniform(-1, 1)

    def draw(self, canvas, color='gray'):
        canvas.create_line(self.from_neuron.x, self.from_neuron.y, self.to_neuron.x, self.to_neuron.y, fill=color)
        mid_x = (self.from_neuron.x + self.to_neuron.x) / 2
        mid_y = (self.from_neuron.y + self.to_neuron.y) / 2
        canvas.create_text(mid_x, mid_y, text=f"{self.value:.2f}", font=("Arial", 8))


class NeuralNetwork:
    def __init__(self, layers, activation_func, ui):
        self.layers = layers
        self.activation_func = activation_func
        self.ui = ui
        self.neurons = []
        self.weights = []
        self.create_network()

    def create_network(self):
        num_layers = len(self.layers)
        layer_width = (self.ui.canvas_width - 200) / (num_layers - 1 if num_layers > 1 else 1)
        for l_index, num_neurons in enumerate(self.layers):
            layer_neurons = []
            layer_height = (self.ui.canvas_height - 200) / (num_neurons - 1 if num_neurons > 1 else 1)
            for n_index in range(num_neurons):
                x = 100 + l_index * layer_width
                y = 100 + n_index * layer_height
                neuron = Neuron(l_index, n_index, x, y)
                layer_neurons.append(neuron)
            self.neurons.append(layer_neurons)

        self.weights = []
        for l in range(len(self.neurons) - 1):
            for from_neuron in self.neurons[l]:
                for to_neuron in self.neurons[l + 1]:
                    self.weights.append(Weight(from_neuron, to_neuron))

    def activation(self, x):
        if self.activation_func == "sigmoid":
            return 1 / (1 + math.exp(-x))
        elif self.activation_func == "tanh":
            return math.tanh(x)
        elif self.activation_func == "relu":
            return max(0, x)

    def activation_derivative(self, output):
        if self.activation_func == "sigmoid":
            return output * (1 - output)
        elif self.activation_func == "tanh":
            return 1 - output ** 2
        elif self.activation_func == "relu":
            return 1 if output > 0 else 0

    def forward(self, inputs):
        # Clear canvas to redraw everything
        self.ui.canvas.delete("all")

        # Input layer
        for i, value in enumerate(inputs):
            self.neurons[0][i].output = value

        # Hidden and output layers
        for l in range(1, len(self.layers)):
            for neuron in self.neurons[l]:
                total_input = neuron.bias
                for weight in self.weights:
                    if weight.from_neuron in self.neurons[l - 1] and weight.to_neuron == neuron:
                        total_input += weight.from_neuron.output * weight.value
                neuron.output = self.activation(total_input)

        # Redraw network with updated outputs
        # Draw weights once (without changing colors), to keep it quick
        for weight in self.weights:
            weight.draw(self.ui.canvas)

        # Draw updated neurons
        for layer in self.neurons:
            for neuron in layer:
                neuron.draw(self.ui.canvas)

        self.ui.canvas.update()
        return [neuron.output for neuron in self.neurons[-1]]

    def backward(self, targets, learning_rate):
        # Output layer
        for i, neuron in enumerate(self.neurons[-1]):
            error = targets[i] - neuron.output
            neuron.delta = error * self.activation_derivative(neuron.output)

        # Hidden layers
        for l in reversed(range(1, len(self.layers) - 1)):
            for neuron in self.neurons[l]:
                error = sum(weight.to_neuron.delta * weight.value for weight in self.weights if weight.from_neuron == neuron)
                neuron.delta = error * self.activation_derivative(neuron.output)

        # Update weights and biases
        for weight in self.weights:
            change = learning_rate * weight.from_neuron.output * weight.to_neuron.delta
            weight.value += change

        for layer in self.neurons[1:]:
            for neuron in layer:
                neuron.bias += learning_rate * neuron.delta


class NeuralNetworkUI:
    def __init__(self):
        self.app = tk.Tk()
        self.app.geometry("1200x800")
        self.app.title("Neural Network Visualization")

        self.canvas_width = 800
        self.canvas_height = 800

        self.data_preprocessor = DataPreprocessor()
        self.network = None
        self.activation_func = "sigmoid"
        self.learning_rate = 0.5
        self.epochs = 10

        self.create_ui()

    def create_ui(self):
        control_panel = Frame(self.app)
        control_panel.pack(side="right", padx=10, pady=10)

        Label(control_panel, text="Layers (comma-separated):").pack()
        self.layer_entry = Entry(control_panel)
        self.layer_entry.pack()
        self.layer_entry.insert(0, "4,6,1")

        Label(control_panel, text="Learning Rate:").pack()
        self.lr_entry = Entry(control_panel)
        self.lr_entry.pack()
        self.lr_entry.insert(0, "0.5")

        Label(control_panel, text="Epochs:").pack()
        self.epochs_entry = Entry(control_panel)
        self.epochs_entry.pack()
        self.epochs_entry.insert(0, "10")

        Label(control_panel, text="Activation Function:").pack()
        self.activation_var = StringVar(value="sigmoid")
        for func in ["sigmoid", "tanh", "relu"]:
            Radiobutton(control_panel, text=func.capitalize(), variable=self.activation_var, value=func).pack()

        Button(control_panel, text="Load CSV", command=self.load_csv).pack()
        Button(control_panel, text="Generate Network", command=self.generate_network).pack()
        Button(control_panel, text="Train Network", command=self.start_training).pack()
        Button(control_panel, text="Test Network", command=self.start_testing).pack()

        self.canvas = Canvas(self.app, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(side="left", padx=10, pady=10)

    def load_csv(self):
        file_path = askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data_preprocessor.load_csv(file_path)
            self.data_preprocessor.preprocess()
            self.data_preprocessor.split_data()
            print(f"Loaded dataset from {file_path}")

    def generate_network(self):
        layers = [int(x) for x in self.layer_entry.get().split(",")]
        self.learning_rate = float(self.lr_entry.get())
        self.epochs = int(self.epochs_entry.get())
        self.activation_func = self.activation_var.get()
        self.network = NeuralNetwork(layers, self.activation_func, self)
        self.draw_network()

    def draw_network(self):
        self.canvas.delete("all")
        for weight in self.network.weights:
            weight.draw(self.canvas)
        for layer in self.network.neurons:
            for neuron in layer:
                neuron.draw(self.canvas)
        self.canvas.update()

    def compute_accuracy(self, data):
        correct = 0
        for (inputs, targets) in data:
            outputs = self.network.forward(inputs)
            prediction = 1 if outputs[0] > 0.5 else 0
            if prediction == targets[0]:
                correct += 1
        accuracy = (correct / len(data)) * 100 if data else 0
        return accuracy

    def start_training(self):
        if not self.network:
            print("Generate the network first.")
            return

        training_data = self.data_preprocessor.training_data
        print("Training started...")
        for epoch in range(self.epochs):
            total_loss = 0
            random.shuffle(training_data)
            for (inputs, targets) in training_data:
                # Forward pass (UI shows changing outputs)
                outputs = self.network.forward(inputs)
                # Calculate loss (MSE)
                loss = sum((targets[j] - outputs[j]) ** 2 for j in range(len(targets))) / len(targets)
                total_loss += loss
                # Backward pass
                self.network.backward(targets, self.learning_rate)
            avg_loss = total_loss / len(training_data)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        # After all epochs
        final_train_acc = self.compute_accuracy(training_data)
        print(f"Training completed.\nFinal Training Accuracy: {final_train_acc:.2f}%")

    def start_testing(self):
        if not self.network:
            print("Generate the network first.")
            return

        testing_data = self.data_preprocessor.testing_data
        testing_acc = self.compute_accuracy(testing_data)
        print(f"Testing completed.\nTesting Accuracy: {testing_acc:.2f}%")

    def run(self):
        self.app.mainloop()


if __name__ == "__main__":
    app = NeuralNetworkUI()
    app.run()