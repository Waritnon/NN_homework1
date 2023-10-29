import numpy as np

##กำหนดที่อยึ่ไฟล์ cross.txt
filename = 'C:/Users/warit/OneDrive/Desktop/ci1/cross.txt'

##กำหนดขนาดของ input, hidden, output
input_size = 2
hidden_size = 8
output_size = 2

##กำหนดค่า epochs,mse
target_epochs = 50000  
mean_squared_error = 0.00001  

##สุ่มค่า weight และ bias
weights_input_to_hidden = np.random.randn(hidden_size, input_size)
velocity_input_to_hidden = np.random.randn(hidden_size, input_size)
weights_hidden_to_output = np.random.randn(output_size, hidden_size)
velocity_hidden_to_output = np.random.randn(output_size, hidden_size)
bias_hidden = np.random.randn(hidden_size, 1)
velocity_bias_hidden = np.random.randn(hidden_size, 1)
bias_output = np.random.randn(output_size, 1)
velocity_bias_output = np.random.randn(output_size, 1)

##กำหนดค่า learning_rate และ momentum_rates
learning_rates = [0.025]
momentum_rates = [0.1]

##ฟังก์ชันอ่านไฟล์
def read_file(filename):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            label = lines[i].strip()
            x, y = map(float, lines[i + 1].split())
            i += 2
            label_data = (label, (x, y),)
            label_data += tuple(map(int, lines[i].split()))
            i += 1
            data.append(label_data)
    return data

##ฟังวก์ชันแยก train และ test
def separate_test_train(data, train_ratio=0.9):
    np.random.shuffle(data)
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

##ฟังวก์ชันแยก input และ output
def seperate_input_output(data):
    input_data = np.array([item[1] for item in data])
    output_data = np.array([item[2:] for item in data])
    return input_data, output_data

##ฟังก์ชัน sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

##ฟังก์ชันการ feed forward
def forward_propagation(input_data, weights_input_to_hidden, bias_hidden, weights_hidden_to_output, bias_output):
    hidden = sigmoid(np.dot(weights_input_to_hidden, input_data.T) + bias_hidden)
    output = sigmoid(np.dot(weights_hidden_to_output, hidden) + bias_output)
    return hidden, output

##ฟังก์ชันอัพเดต weight ของ hidden กับ input
def update_input_hidden_weights(input_data, hidden_gradient, learning_rate, momentum_rate):
    global weights_input_to_hidden, bias_hidden, velocity_input_to_hidden, velocity_bias_hidden
    velocity_input_to_hidden = (momentum_rate * velocity_input_to_hidden) + (learning_rate * np.dot(hidden_gradient, input_data) / len(input_data))
    weights_input_to_hidden += velocity_input_to_hidden
    velocity_bias_hidden = (momentum_rate * velocity_bias_hidden) + (learning_rate * np.mean(hidden_gradient, axis=1, keepdims=True))
    bias_hidden += velocity_bias_hidden

##ฟังก์ชันอัพเดต weight ของ hidden กับ output
def update_hidden_output_weights(hidden, output_gradient, learning_rate, momentum_rate):
    global weights_hidden_to_output, bias_output, velocity_hidden_to_output, velocity_bias_output
    velocity_hidden_to_output = (momentum_rate * velocity_hidden_to_output) + (learning_rate * np.dot(output_gradient, hidden.T) / len(hidden))
    weights_hidden_to_output += velocity_hidden_to_output
    velocity_bias_output = (momentum_rate * velocity_bias_output) + (learning_rate * np.mean(output_gradient, axis=1, keepdims=True))
    bias_output += velocity_bias_output

##ฟังก์ชัน train
def train_neural_network(input_data, output_data, weights_input_to_hidden, bias_hidden, velocity_input_to_hidden, velocity_bias_hidden, weights_hidden_to_output, bias_output, velocity_hidden_to_output, velocity_bias_output, target_epochs, mean_squared_error, learning_rate, momentum_rate):
    for epoch in range(target_epochs):
        hidden, output = forward_propagation(input_data, weights_input_to_hidden, bias_hidden, weights_hidden_to_output, bias_output)
        output_error = output_data - output.T
        output_gradient = output_error.T * sigmoid_derivative(output)
        update_hidden_output_weights(hidden, output_gradient, learning_rate, momentum_rate)
        hidden_error = np.dot(weights_hidden_to_output.T, output_gradient)
        hidden_gradient = hidden_error * sigmoid_derivative(hidden)
        update_input_hidden_weights(input_data, hidden_gradient, learning_rate, momentum_rate)
        error = np.mean(output_error**2, axis=0)
        if epoch % 10000 == 0:
            print(f"Epoch: {epoch + 10000}, Error: {error}")
        if np.all(error <= mean_squared_error):
            break

##ฟังก์ชันคำนวน accuracy
def calculate_accuracy(TP, TN, FP, FN):
    total_predictions = TP + TN + FP + FN
    if total_predictions == 0:
        return 0.0
    accuracy = ((TP + TN) / total_predictions) * 100
    return accuracy

##ทำการสั่งอ่านไฟล์ และทำการแยกข้อมูลสำหรับ test หรือ train
data = read_file(filename)
training_data, testing_data = separate_test_train(data)

for lr in learning_rates:
    for momentum in momentum_rates:
        print(f"learning rate = {lr}  momentum = {momentum}")
        input_train, output_train = seperate_input_output(training_data)
        train_neural_network(input_train, output_train, weights_input_to_hidden, bias_hidden,velocity_input_to_hidden, velocity_bias_hidden, weights_hidden_to_output,bias_output, velocity_hidden_to_output, velocity_bias_output,target_epochs, mean_squared_error, lr, momentum)
        input_test, output_test = seperate_input_output(testing_data)
        Actual = output_test
        _, Predict = forward_propagation(input_test, weights_input_to_hidden, bias_hidden, weights_hidden_to_output, bias_output)
        Predict = np.transpose(Predict)
        threshold = 0.5
        predicted = (Predict[:, 1] > threshold).astype(int)
        confusion_matrix = np.zeros((2, 2), dtype=int)
        for i in range(2):
            for j in range(2):
                confusion_matrix[i, j] = np.sum((Actual[:, i] == 1) & (predicted == j))
        TP = confusion_matrix[1, 1]
        TN = confusion_matrix[0, 0]
        FP = confusion_matrix[0, 1]
        FN = confusion_matrix[1, 0]
        Accuracy = calculate_accuracy(TP, TN, FP, FN)
        
        ##การแสดงผล confusion matrix และความแม่นยำ
        print("\nResults:")
        print("Confusion Matrix:")
        print(confusion_matrix)
        print(f"Accuracy: {Accuracy:.2f}%")
