import numpy
import scipy.special


class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set numbers
        self.train_times = 0
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.activation_function = lambda x: scipy.special.expit(x)

        # 输入层和隐藏层的链接权重, 初始是random, 正态分布
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 计算给隐藏层的
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs

        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 更新隐藏层和输出层权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # 更新输入层和隐藏层的权重
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        self.train_times += 1
        if self.train_times % 100 == 0:
            print(self.train_times)


    def query(self, input_list):
        # 给隐藏层的
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 给输出层的, 应用这层的链接权重
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


if __name__ == '__main__':
    import time
    time_start = time.time()
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    learning_rate = 0.3

    foo = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    with open('mnist_train.csv', 'r') as f_in:
        data_list = f_in.readlines()

    for record in data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        foo.train(inputs, targets)


    pass
    with open('mnist_test.csv') as f_in:
        result_list = f_in.readlines()

    correct_count = 0
    for _ in result_list:
        all_values = _.split(',')
        correct = int(all_values[0])
        print(f'{correct} correct')
        rs = foo.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
        label = numpy.argmax(rs)
        print(f'{label} answer')
        if label == correct:
            correct_count += 1

    rate = correct_count / int(len(result_list))
    print(f'Total correct rate is {rate}')
    time_stop = time.time()
    print(f'Total time cost is {time_stop - time_start}')
