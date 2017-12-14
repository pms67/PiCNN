#ifndef CNN_H
#define CNN_H

#include "Matrix.h"
#include "Layer.h"
#include "lInput.h"
#include "lConv.h"
#include "lPool.h"
#include "lFlatten.h"
#include "lDense.h"
#include "DataHandler.h"

class CNN {
	
	private:
	
		int feature_maps;
		int in_rows;
		int in_cols;
		int out_rows;
		int out_cols;
		
		int nlayers;
		
	
	public:
			
		std::vector<Layer*> layers;
				
		//Constructor
		CNN( int in_rows, int in_cols, int feature_maps ) {
						
			//Set dimensions
			this->in_rows = in_rows;
			this->in_cols = in_cols;
			this->feature_maps = feature_maps;
			out_rows = in_rows;
			out_cols = in_cols;
			
			//Add input layer
			layers.push_back(new lInput(feature_maps, in_rows, in_cols));			
			nlayers = 1;
			
		}
		
		//Properties
		int getLayers() { return nlayers; }
		int getOutRows() { return out_rows; }
		int getOutCols() { return out_cols; }
		int getFeatureMaps() { return feature_maps; }
		
		//Functions
		void addConv(int k_size, std::string activation) {
					
			layers.push_back(new lConv(feature_maps, layers[nlayers-1]->out_rows, layers[nlayers-1]->out_cols, k_size, activation));
				
			nlayers++;
			out_rows = layers[nlayers-  1]->out_rows;
			out_cols = layers[nlayers - 1]->out_cols;
			
			return;
			
		}
		
		void addPool(int p_size) {
							
			layers.push_back(new lPool(feature_maps, layers[nlayers-1]->out_rows, layers[nlayers-1]->out_cols, p_size));
			
			nlayers++;
			out_rows = layers[nlayers-  1]->out_rows;
			out_cols = layers[nlayers - 1]->out_cols;
			
			return;
			
		}
		
		void addFlatten() {
										
			layers.push_back(new lFlatten(feature_maps, layers[nlayers-1]->out_rows, layers[nlayers-1]->out_cols));
			
			nlayers++;
			out_rows = layers[nlayers-  1]->out_rows;
			out_cols = layers[nlayers - 1]->out_cols;
			
			return;
						
		}
		
		void addDense(int out_size) {
							
			layers.push_back(new lDense(layers[nlayers-1]->out_rows, out_size));
			
			nlayers++;
			out_rows = layers[nlayers-  1]->out_rows;
			out_cols = layers[nlayers - 1]->out_cols;
			
			return;
			
		}
		
		Tensor feedforward( Tensor in ) {
			
			for (int i = 0; i < nlayers; i++) {
				
				layers[i]->feedforward(in);
				in = layers[i]->out.copy();
				
			}
			
			return layers[nlayers-1]->out;
			
		}
		
		double train( Tensor in, Tensor target, double lrate, double mom ) {
			
			//Feedforward inputs
			feedforward(in);
					
			//Get deltas at output and cost
			Tensor delta(1, out_rows, out_cols);
								
			double cost = getCost(target, delta);
			
			for (int i = nlayers - 1; i > 0; i--) {
				
				delta = layers[i]->feedback(delta).copy();
				
				//Update weights
				layers[i]->updateweights(lrate, mom);
				
			}
			
			return cost;
			
		}
		
		double train (TensorArray in, TensorArray target, double lrate, double mom, int max_epochs, double min_err, int batch_size) {
			
			DataHandler dh;
			int epoch = 1;
			double err = pow(10, 5);
			std::default_random_engine generator;
			std::uniform_real_distribution<double> distribution(0, in.size());
			
			generator.seed(time(0));
	
			std::cout << "Training (Max. Epochs: " << max_epochs << ", Batch Size: " << batch_size << ", Min. Error: " << min_err << ", Momentum: " << mom << ")...\n";
				
			while (epoch <= max_epochs && err > min_err) {
				
				std::clock_t begin = std::clock();
				err = 0.0;
				
				for (int i = 0; i < batch_size; i++) {
					
					int n = int(distribution(generator));
					err += train(in[n], target[n], lrate, mom);
					
				}
				
				err /= batch_size;
				
					
				int num_correct = 0;
				
				for (int i = 0; i < in.size(); i++) {
					
					Tensor output = feedforward(in[i]);
					num_correct += (dh.compare_onehot(output, target[i]) ? 1 : 0);
				
				}
				
				std::clock_t end = std::clock(); 
				
				double val_acc = double(num_correct * 100) / in.size();
				
				lrate = dh.getLearningRate(val_acc, 0.01, 0.1, 30, 85);
						
				std::cout << "Epoch: " << epoch << "/" << max_epochs << " | Training Error: " << err << " | Validation Accuracy: " << val_acc <<"% | Learning Rate: " << lrate << " | Time Taken: " << double(end - begin) / CLOCKS_PER_SEC << "s\n";
				
				epoch++;
		
			}
					
			return err;			
			
		}		
						
		double getCost( Tensor target ) {
			
			double cost = 0.0;
			
			for (int i = 0; i < out_rows; i++) {
				
				for (int j = 0; j < out_cols; j++) {
					
					cost += pow(layers[nlayers-1]->out(0, i, j) - target(0, i, j), 2);
					
				}
				
			}
			
			cost /= (2 * out_rows * out_cols);
			
			return cost;
			
		}
		
		double getCost( Tensor target, Tensor & delta ) {
			
			double cost = 0.0;
						
			for (int i = 0; i < out_rows; i++) {
				
				for (int j = 0; j < out_cols; j++) {
					
					delta(0, i, j) = (layers[nlayers - 1]->out(0, i, j) - target(0, i, j));
					cost += pow(layers[nlayers - 1]->out(0, i, j) - target(0, i, j), 2);
					
				}
				
			}
			
			cost /= (2 * out_rows * out_cols);
			
			return cost;
			
		}
		
		void print() {
						
			std::stringstream s;
						
			s << "[Network] In: " << in_rows << "x" << in_cols << " | Out: " << out_rows << "x" << out_cols << " | Feature Maps: " << feature_maps << std::endl;
								
		
			for (int i = 0; i < nlayers; i ++)				
				s << "[Layer " << (i + 1) << "]: Type: " << layers[i]->getType() << " | Out: " << layers[i]->getDim() << "x" << layers[i]->getRows() << "x" << layers[i]->getCols() << std::endl;
			
			std::cout << s.str();
			
			return;
			
		}
						
};

#endif
