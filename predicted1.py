import numpy as np
##Node number
input_s = 8
hidden_s = 8
output_s = 1

##generate weight input, hidden current, and past 
w_input_to_hidden = np.random.randn(hidden_s, input_s)
lastw_input_hidden = np.random.randn(hidden_s, input_s)

w_hidden_to_output = np.random.randn(output_s, hidden_s)
lastw_hidden_output = np.random.randn(output_s, hidden_s)

b_hidden = np.random.randn(hidden_s, 1)
lastb_hidden = np.random.randn(hidden_s, 1)

b_output = np.random.randn(output_s, 1)
lastb_output = np.random.randn(output_s, 1)

##sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

##sigmoid derivative function
def sigmoid_derivative(x):
    return x * (1 - x)

##readfile store on array
def read_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            columns = line.strip().split()  
            numeric_columns = [int(col) for col in columns]
            data.append(numeric_columns)
    data_array = np.array(data)
    return data_array

##cross validation 10%
def cross_validation_split(data_array, train_percent=0.1):
    np.random.shuffle(data_array) 
    split_index = int(len(data_array) * train_percent)
    train_data = data_array[:split_index]
    test_data = data_array[split_index:]
    return train_data, test_data

##normalize_data to 0-1
def normalize_data(data, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.min(data, axis=0)
    if max_val is None:
        max_val = np.max(data, axis=0)
    
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data, min_val, max_val

##inverse normalize to normal data
def inverse_normalize_data(normalized_data, min_val, max_val):
    original_data = normalized_data * (max_val - min_val) + min_val
    return original_data

##separate 8 columns to input, and the last column is the output
def separate_input_output(data):
    input_data = data[:, :8]  
    output_data = data[:, 8] 
    return input_data, output_data

##forward propagation function
def forward_propagation(input_data, w_input_to_hidden, b_hidden, w_hidden_to_output, b_output):
    hidden_input = np.dot(w_input_to_hidden, input_data.T) + b_hidden
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(w_hidden_to_output, hidden_output) + b_output
    output_output = sigmoid(output_input)
    return hidden_output, output_output

##update weights at input and hidden layers
def update_weights_input_hidden(input_data, hidden_output, output_output, target_output, w_hidden_to_output, w_input_to_hidden, learning_rate):
    output_error = target_output - output_output
    output_delta = output_error * sigmoid_derivative(output_output)
    hidden_error = np.dot(w_hidden_to_output.T, output_delta)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
    w_hidden_to_output += learning_rate * np.dot(output_delta, hidden_output.T)
    w_input_to_hidden += learning_rate * np.dot(hidden_delta, input_data)

##update weights at hidden and output layers
def update_weights_hidden_output(hidden_output, output_output, target_output, w_hidden_to_output, b_output, learning_rate):
    output_error = target_output - output_output
    output_delta = output_error * sigmoid_derivative(output_output)
    w_hidden_to_output += learning_rate * np.dot(output_delta, hidden_output.T)
    b_output += learning_rate * np.sum(output_delta, axis=1, keepdims=True)

##train neural network function
def train_neural_network(input_data, target_output, target_epochs, mean_squared_error, learning_rate, momentum_rate):
    for epochs in range(target_epochs):
        hidden_output, output_output = forward_propagation(input_data, w_input_to_hidden, b_hidden, w_hidden_to_output, b_output)
        output_error = target_output - output_output
        output_gradient = output_error * sigmoid_derivative(output_output)
        update_weights_hidden_output(hidden_output, output_output, target_output, w_hidden_to_output, b_output, learning_rate)
        hidden_error = np.dot(w_hidden_to_output.T, output_gradient)
        hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)
        update_weights_input_hidden(input_data, hidden_output, output_output, target_output, w_hidden_to_output, w_input_to_hidden, learning_rate)
        error = np.mean(output_error**2)
        if epochs % 10000 == 0:
            print(f"Epoch loop: {epochs + 10000}, Error: {error}")
            display_results(target_output, output_output)

        if error <= mean_squared_error:
            print(f"Training completed")
            break
    
    # Calculate accuracy using inverse normalization
    predicted_output = inverse_normalize_data(output_output.T, min_val[8], max_val[8])
    target_output = inverse_normalize_data(target_output.T, min_val[8], max_val[8])
    accuracy = 100 - np.mean(np.abs((target_output - predicted_output) / target_output) * 100)
    print(f"************Accuracy = {accuracy} % **************")

##Function show 
def display_results(target_output, output_output):
    print("Actual Output    Predict Output      Error")
    for i in range(min(len(target_output), len(output_output))):
        target = inverse_normalize_data(target_output[i], min_val[8], max_val[8])
        predicted = inverse_normalize_data(output_output[i], min_val[8], max_val[8])
        error = abs(target - predicted)
        if np.isscalar(target):
            target = np.array([target])
        if np.isscalar(predicted):
            predicted = np.array([predicted])
        if np.isscalar(error):
            error = np.array([error])
        print("    {:.2f}           {:.2f}          {:.2f}%".format(target[0], predicted[0], error[0]))
        print("\n")

#varaible and usage
data_array = read_file("C:/Users/warit/OneDrive/Desktop/ci1/file.txt")
normalized_data, min_val, max_val = normalize_data(data_array)
input_data, output_data = separate_input_output(normalized_data)
train_data, test_data = cross_validation_split(data_array, train_percent=0.1)
learning_rate = 0.01  
momentum_rate = 0.2
target_output = output_data 
target_epochs = 50000
mean_squared_error = 0.001 

# Train function
train_neural_network(input_data, target_output, target_epochs, mean_squared_error, learning_rate, momentum_rate)
