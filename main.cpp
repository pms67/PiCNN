#include "CNN.h"

using namespace std;

int main() {

	cout << "Loading training data and labels and normalising...\n";
	
	DataHandler dh;
	TensorArray train_data = dh.normalise_minmax(dh.reshape(dh.readCSV("mnist_train_data_TINY.csv", ","), 28, 28), 0, 255);
	TensorArray train_labels = dh.onehot(dh.readCSV("mnist_train_labels_TINY.csv", ","), 10);	
	
	cout << "Creating neural network...\n";
	
	CNN c(28, 28, 3);
	
	c.addConv(3, "relu");
	c.addPool(2);
	c.addConv(3, "relu");
	c.addPool(2);
	c.addFlatten();
	c.addDense(25);
	c.addDense(10);
	
	cout << "Network structure:\n";
	c.print();
	
	int batch_size = 100;
	int max_epochs = pow(10, 2);
	double min_error = pow(10, -3);
	double lrate = 0.1;
	double mom = 0.01;
	
	c.train(train_data, train_labels, lrate, mom, max_epochs, min_error, batch_size);
	
	cout << "Validation...\n";
	
	int num_correct = 0;
	for (int i = 0; i < train_data.size(); i++) {
		
			Tensor output = c.feedforward(train_data[i]);
			num_correct += (dh.compare_onehot(output, train_labels[i]) ? 1 : 0);
		
		}
	
	cout << "Correct: " << num_correct << "/" << train_data.size() << endl;
	cout << "Done!";

	return 0;
		
}


