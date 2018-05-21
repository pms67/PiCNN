#include "CNN.h"

using namespace std;

void test();

int main() {

	test();
	return 0;				
										
}

void test() {
	
	cout << "Loading training data and labels and normalising...\n";
	
	DataHandler dh;
	TensorArray train_data = dh.normalise_minmax(dh.reshape(dh.readCSV("mnist_train_data_TINY.csv", ","), 28, 28), 0, 255);
	TensorArray train_labels = dh.onehot(dh.readCSV("mnist_train_labels_TINY.csv", ","), 10);	
	
	cout << "Creating neural network...\n";
	
	CNN c(28, 28, 5);
	
	c.addConv(3);
	c.addPool(2);
	c.addFlatten();
	c.addDense(10);
	
	cout << "Network structure:\n";
	c.print();
	
	cout << "Initial Weights of conv layer:\n";
	c.printWeights(1);
	
	int batch_size = 100;
	int max_epochs = pow(10, 1);
	double min_error = pow(10, -3);
	double lrate = 0.1;
	
	int epoch = 1;
	double err = pow(10, 3);
	default_random_engine generator;
	uniform_real_distribution<double> distribution(0, train_data.size());
			
	generator.seed(time(0));
	
	cout << "Training (Max. Epochs: " << max_epochs << ", Batch Size: " << batch_size << ", Min. Error: " << min_error << ", Learning Rate: " << lrate << ")...\n";
	
	while (epoch <= max_epochs && err > min_error) {
		
		clock_t begin = clock();
		err = 0.0;		
	
		for (int i = 0; i < batch_size; i++) {
			
			int n = int(distribution(generator));
		
			err += c.train(train_data[n], train_labels[n], lrate);
			
		}
		
		err /= batch_size;
		
		int num_correct = 0;
	
		for (int i = 0; i < train_data.size(); i++) {
			
			Tensor output = c.feedforward(train_data[i]);
			num_correct += (dh.compare_onehot(output, train_labels[i]) ? 1 : 0);
		
		}
		
		clock_t end = clock();
		
		double val_acc = double(num_correct * 100) / train_data.size();
		
		lrate = dh.getLearningRate(val_acc, 0.001, 0.15, 30, 85);
				
		cout << "Epoch: " << epoch << "/" << max_epochs << " | Training Error: " << err << " | Validation Accuracy: " << val_acc <<"% | Learning Rate: " << lrate << " | Time Taken: " << double(end - begin) / CLOCKS_PER_SEC << "s\n";
		
		epoch++;		
		
		
	}
	
	cout << "Validation...\n";
	
	int num_correct = 0;
	for (int i = 0; i < train_data.size(); i++) {
		
			Tensor output = c.feedforward(train_data[i]);
			num_correct += (dh.compare_onehot(output, train_labels[i]) ? 1 : 0);
		
		}
	
	cout << "Correct: " << num_correct << "/" << train_data.size() << endl;
	
	cout << "Final Weights of conv layer:\n";
	c.printWeights(1);
		
	cout << "Done!";
																						
	return;
	
}
