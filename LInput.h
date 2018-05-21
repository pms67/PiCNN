#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "Layer.h"

class LInput : public Layer {
		
	public:
		
		//Constructor
		LInput( int out_dim, int in_rows, int in_cols ) {
			
			//Set dimensions
			this->in_rows = in_rows;
			this->in_cols = in_cols;
			this->in_dim = out_dim; //This could be different! See feedforward()!
						
			this->out_rows = in_rows;
			this->out_cols = in_cols;
			this->out_dim = out_dim; //Number of feature maps
			
			in.resize(in_dim, in_rows, in_cols);
			out.resize(out_dim, out_rows, out_cols);
			
		}
		
		//Properties
		char getType() { return 'i'; }
		
		//Functions
		Tensor feedforward( Tensor in ) {
						
			if (in.getDim() < in_dim) { //Need to artifically increase input dimensions to match feature map dimensions
			
				for (int d = 0; d < in_dim; d++) {
				
					for (int i = 0; i < in_rows; i++) {
					
						for (int j = 0; j < in_cols; j++) {
						
							if (d < in.getDim()) { //Within bounds
								
								this->in(d, i, j) = in(d, i, j);
								
							} else { //Out of bounds: copy first dimension into this one.
							
								this->in(d, i, j) = in(0, i,  j);
							
							}
						
						}
						
					}
					
				}							
				
				this->out = this->in.copy();
				
				return out;								
				
			} else { //Input dimensions are greater or equal to the feature map dimensions
				
				this->in = in.copy();
				this->out = in.copy();
				
				return out;
				
			}		
			
		}
		
		Tensor feedback( Tensor delta ) { return NULL; }
		
		void updateweights( float rate ) { return; }
	
	
};

#endif
