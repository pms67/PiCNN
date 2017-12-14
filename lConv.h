#ifndef LCONV_H
#define LCONV_H

#include "Layer.h"
#include "Activation.h"
#include <cmath>

class lConv : public Layer {
	
	private:
		
		//Properties
		int w_size;
		
		//Weights and bias
		Tensor weights;
		Tensor bias;
		
		//Gradients
		Tensor dCdX;
		Tensor dCdW;
		Tensor dCdB;
		
		Tensor prev_dCdW;
		Tensor prev_dCdB;
				
		//Activation
		std::string act;
		
	public:
							
		//Constructor
		lConv(int in_dim, int in_rows, int in_cols, int w_size, std::string activation) {
			
			//Set dimensions
			this->in_dim = in_dim;
			this->out_dim = in_dim;
			this->in_rows = in_rows;
			this->in_cols = in_cols;
			this->w_size = w_size;
			
			out_rows = in_rows - w_size + 1;
			out_cols = in_cols - w_size + 1;
			
			//Redimension matrices
			in.resize(in_dim, in_rows, in_cols);
			out.resize(in_dim, out_rows, out_cols);
			weights.resize(in_dim, w_size);
			bias.resize(in_dim, 1, 1);
			dCdX.resize(in_dim, in_rows, in_cols);
			dCdW.resize(in_dim, w_size);
			dCdB.resize(in_dim, 1, 1);
			prev_dCdW.resize(in_dim, w_size);
			prev_dCdB.resize(in_dim, 1, 1);
			
			//Initialise weights and bias
			weights.randn(0.0, 1.0);
			bias.rand(0.0, 1.0);
			
			//Set activation
			act = activation;
							
		}
		
		//Properties
		char getType() { return 'c'; }
		
		//Functions		
		Tensor feedforward( Tensor in ) {
						
			this->in = in.copy();
		
			for (int d = 0; d < in_dim; d++) {
			
				for (int m = 0; m < out_rows; m++) {
		
		    		for (int n = 0; n < out_cols; n++) {
		
		    			out(d, m, n) = bias(d, 0 , 0); //Add bias
		
		    			for (int i = 0; i < w_size; i++) {
		
		        			for (int j = 0; j < w_size; j++) {
		
		          				//Check bounds and convolve
		          				if (m - i >= 0 && n - j >= 0 && m - i < in_rows && n - j < in_cols)
			        				out(d, m, n) += in(d, m - i, n - j) * weights(d, i, j);
		
		        			}
		
						}
		
		      			//Apply non-linearity (ReLU)
		      			if (out(d, m, n) < 0) 
		        			out(d, m, n) = out(d, m, n);		        			       			
		
		    		}
		
		  		}
		  	
		  	}
		  			
		  	return out;
		
		}
		
		Tensor feedback( Tensor delta ) {
			
			//Calculate changes in gradients
			prev_dCdW = dCdW.copy();
			prev_dCdB = dCdB.copy();			
			
			//Reset gradients
			dCdX.clear();
			dCdW.clear();
  			dCdB.clear();
  			
			for (int d = 0; d < in_dim; d++) {
			
  				for (int m = 0; m < out_rows; m++) {

    				for (int n = 0; n < out_cols; n++) {

      					if (out(d, m, n) > 0) { //ReLu derivative property

        					//Bias
        					dCdB(d, 0, 0) += delta(d, m, n);

        					//Deltas
        					for (int a = 0; a < in_rows; a++) {

          						for (int b = 0; b < in_cols; b++) {

            						if (m - a >= 0 && n - b >= 0 && m - a < w_size && n - b < w_size)
              							dCdX(d, a, b) += delta(d, m, n) * weights(d, m - a, n - b);

          						}

        					}

        					//Weights
        					for (int a = 0; a < w_size; a++) {

          						for (int b = 0; b < w_size; b++) {

            						if (m - a >= 0 && n - b >= 0 && m - a < in_rows && n - b < in_cols)
	          							dCdW(d, a, b) += delta(d, m , n) * in(d, m - a, n - b);

          						}

        					}

      					}

    				}

  				}
  			
  			}

  			return dCdX;

		}
		
		void updateweights( float rate, float mom ) {
			
			for (int d = 0; d < in_dim; d++) {
				
				//Update bias
				bias(d, 0, 0) -= rate * dCdB(d, 0, 0) + mom * (dCdB(d, 0, 0) - prev_dCdB(d, 0, 0));
				
				//Update kernel
				for (int i = 0; i < w_size; i++)
					for (int j = 0; j < w_size; j++)
						weights(d, i, j) -= rate * dCdW(d, i, j) + mom * (dCdW(d, i, j) - prev_dCdW(d, i, j));
			
			}
					
			return;			
			
		}
			
};

#endif
