#ifndef LFLATTEN_H
#define LFLATTEN_H

#include "Layer.h"

class lFlatten : public Layer {
	
	private:
											
		//Gradients
		Tensor dCdX;
		
	public:
				
		//Constructor
		lFlatten( int in_dim, int in_rows, int in_cols ) {
			
			//Set dimensions
			this->in_dim = in_dim;
			this->in_rows = in_rows;
			this->in_cols = in_cols;
			
			this->out_dim = 1;
			out_rows = in_dim * in_rows * in_cols;
			out_cols = 1;
			
			//Redimension matrices
			in.resize(in_dim, in_rows, in_cols);
			out.resize(out_dim, out_rows, out_cols);
			dCdX.resize(in_dim, in_rows, in_cols);		
										
		}
		
		//Properties
		char getType() { return 'f'; }
		
		//Functions		
		Tensor feedforward( Tensor in ) {
						
			this->in = in.copy();
			
			int count = 0;
			
			for (int d = 0; d < in_dim; d++) {
			
				for (int i = 0; i < in_rows; i++) {
				
					for (int j = 0; j < in_cols; j++) {
					
						out(0, count, 0) = in(d, i, j);
						count++;
					
					}
				
				}
			
			}
						
			return out;			
				
		}
		
		Tensor feedback( Tensor delta ) {
			
			int count = 0;
			
			for (int d = 0; d < in_dim; d++) {
						
				for (int i = 0; i < in_rows; i++) {
								
					for (int j = 0; j < in_cols; j++) {
					
						dCdX(d, i, j) = delta(0, count, 0);
						count++;
						
					}
				
				}
				
			}
			
			return dCdX;
		
		}
		
		void updateweights( float rate, float mom ) { return; }
		
};

#endif
