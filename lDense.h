#ifndef LDENSE_H
#define LDENSE_H

#include "Layer.h"
#include <cmath>

class lDense : public Layer {
	
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
		lDense(int in_size, int out_size) {
			
			//Set dimensions
			this->in_dim = 1;
			this->in_rows = in_size;
			this->in_cols = 1;
			
			this->out_dim = 1;
			out_rows = out_size;
			out_cols = 1;
			
			//Redimension matrices
			in.resize(in_dim, in_rows, in_cols);
			out.resize(out_dim, out_rows, out_cols);
			weights.resize(in_dim, out_rows, in_rows);
			bias.resize(in_dim, out_rows, out_cols);
			dCdX.resize(in_dim, in_rows, in_cols);
			dCdW.resize(in_dim, out_rows, in_rows);
			dCdB.resize(in_dim, out_rows, out_cols);
			prev_dCdW.resize(in_dim, out_rows, in_rows);
			prev_dCdB.resize(in_dim, out_rows, out_cols);
			
			//Initialise weights and biases
			weights.randn(0.0, 0.1);
			bias.randn(0.0, 1.0);	
													
		}
		
		//Properties
		char getType() { return 'd'; }
		
		//Functions		
		Tensor feedforward( Tensor in ) {
						
			this->in = in.copy();
			
			for (int i = 0; i < out_rows; i++) {
			
				out(0, i, 0) = bias(0, i, 0);
				
				for (int j = 0; j < in_rows; j++) {
					
					out(0, i, 0) += weights(0, i, j) * in(0, j, 0);
					
				}
				
				//Apply non-linearity
				out(0, i, 0) = tanh(out(0, i, 0));
			
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
				
				dCdB(0, i, 0) = delta(0, i, 0) * (1 - pow(out(0, i, 0), 2));
				
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
