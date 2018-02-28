#ifndef SOFTMAX_LAYER
#define SOFTMAX_LAYER

#include "Layer.h"
#include <cmath>

class LSoftmax : public Layer {
	
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
		
	public:		

		//Constructor
		LSoftmax( int in_size, int out_size ) {
			
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
			double expsum = 0.0;
			
			for (int i = 0; i < out_rows; i++) {
			
				in_w(0, i, 0) = bias(0, i, 0);
				
				for (int j = 0; j < in_rows; j++) {
					
					in_w(0, i, 0) += weights(0, i, j) * in(0, j, 0);
					
				}
				
				//Exponentiate
				out(0, i, 0) = exp(in_w(0, i, 0));
				expsum += out(0, i, 0);
			
			}
			
			//Apply softmax normalisation
			for (int i = 0; i < out_rows; i++)
				out(0, i, 0) /= expsum;
						
			return out;			
				
		}
		
		//Kronecker delta
		int kron_delta(int i, int j) {
			
			if (i == j)
				return 1;
			else
				return 0;
			
		}
		
		Tensor feedback( Tensor delta ) {

			prev_dCdW = dCdW.copy();
			prev_dCdB = dCdB.copy();

			dCdX.set(0);
			dCdW.set(0);
			dCdB.set(0);
			
			//Bias
			for (int i = 0; i < out_rows; i++)		
				for (int j = 0; j < out_rows; j++)
					dCdB(0, i, 0) += delta(0, j, 0) * out(0, j, 0) * ( kron_delta(i, j) - out(0, i, 0) );
				
			//Weights	
			for (int i = 0; i < out_rows; i++)
				for (int j = 0; j < in_rows; j++)
					for (int m = 0; m < out_rows; m++)
						for (int n = 0; n < out_rows; n++)
							dCdW(0, i, j) += delta(0, m, 0) * out(0, m, 0) * ( kron_delta(m, n) - out(0, n, 0) ) * in(0, j , 0) * kron_delta(n, i);
			
			//Deltas
			for (int i = 0; i < in_rows; i++)
				for (int j = 0; j < out_rows; j++)
					for (int k = 0; k < out_rows; k++)
						dCdX(0, i, 0) += delta(0, j, 0) * out(0, j, 0) * ( kron_delta(j, k) - out(0, k, 0) ) * weights(0, k, i);
										
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
