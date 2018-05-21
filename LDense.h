#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "Layer.h"

class LDense : public Layer {
	
	private:
							
		//Weights ans Biases
		Tensor weights;
		Tensor bias;
				
		//Gradients
		Tensor dCdX;
		Tensor dCdW;
		Tensor dCdB;
		
		Tensor prev_dCdW;
		Tensor prev_dCdB;
		
		//Activation
		Activation* act;
		
	public:		
	
		//Constructor
		LDense( int in_size, int out_size, char act_func ) {
			
			//Set dimensions
			this->in_dim = 1;
			this->in_rows = in_size;
			this->in_cols = 1;
			
			this->out_dim = 1;
			out_rows = out_size;
			out_cols = 1;
			
			//Redimension matrices
			in.resize(in_dim, in_rows, in_cols);
			in_w.resize(out_dim, out_rows, out_cols);
			out.resize(out_dim, out_rows, out_cols);			
			
			weights.resize(in_dim, out_rows, in_rows);
			bias.resize(in_dim, out_rows, out_cols);
			
			dCdX.resize(in_dim, in_rows, in_cols);
			dCdW.resize(in_dim, out_rows, in_rows);
			dCdB.resize(in_dim, out_rows, out_cols);
			
			prev_dCdW.resize(in_dim, out_rows, in_rows);
			prev_dCdB.resize(in_dim, out_rows, out_cols);
						
			//Set activation function
			switch (act_func) {
				
				case 's':
					act = new Sigmoid();
					break;
					
				case 't':
					act = new Tanh();
					break;
					
				case 'a':
					act = new ArcTan();
					break;
					
				case 'r':
					act = new ReLU();
					break;
					
				case 'l':
					act = new LeakyReLU();
					break;
					
				case 'f':
					act = new FastSigmoid();
					break;
				
				default:
					act = new Sigmoid();
				
			}
													
		}
		
		//Properties
		char getType() { return 'd'; }
		void print() { weights.print();	bias.print(); return; }
		void printGrads() { dCdW.print(); dCdB.print(); return;	}
		
		//Functions	
		//Initialise weights randomly according to a normal distribution
		void initweights( std::default_random_engine generator, double mean, double stddev ) {
			
			std::normal_distribution<double> distribution(mean, stddev);
			
			for (int i = 0; i < out_rows; i++) {
				
				bias(0, i, 0) = distribution(generator);
				
				for (int j = 0; j < in_rows; j++)
					weights(0, i, j) = distribution(generator);
			
			}
			
			return;
			
		}	
				
		Tensor feedforward( Tensor in ) {
			
			//Copy inputs
			this->in = in.copy();
			
			for (int i = 0; i < out_rows; i++) {
			
				in_w(0, i, 0) = bias(0, i, 0);
				
				for (int j = 0; j < in_rows; j++) {
					
					in_w(0, i, 0) += weights(0, i, j) * in(0, j, 0);
					
				}
				
				//Apply non-linearity
				out(0, i, 0) = act->activate(in_w(0, i, 0));
			
			}			
						
			return out;			
				
		}
		
		Tensor feedback( Tensor delta ) {

			prev_dCdW = dCdW.copy();
			prev_dCdB = dCdB.copy();

			dCdX.set(0);
			dCdW.set(0);
			dCdB.set(0);
			
			for (int i = 0; i < out_rows; i++) {
				
				double deriv = act->derivative(in_w(0, i, 0));
				
				dCdB(0, i, 0) = delta(0, i, 0) * deriv;
				
				for (int j = 0; j < in_rows; j++) {
					
					dCdW(0, i, j) = dCdB(0, i, 0) * in(0, j, 0);
					dCdX(0, j, 0) += dCdB(0, i, 0) * weights(0, i, j);
					
				}
				
			}
						
			return dCdX;
		
		}
		
		void updateweights( float rate, float mom ) {
			
			for (int i = 0; i < out_rows; i++) {
				
				bias(0, i, 0) -= rate * dCdB(0, i, 0) + mom * (dCdB(0, i, 0) - prev_dCdB(0, i, 0));
				
				for (int j = 0; j < in_rows; j++) {
					
					weights(0, i, j) -= rate * dCdW(0, i, j) + mom * (dCdW(0, i, j) - prev_dCdW(0, i , j));					
					
				}			
				
			}			
			
			return;
			
		}
			
};

#endif
