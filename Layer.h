#ifndef LAYER_H
#define LAYER_H

#include "Tensor.h"

class Layer {
		
	public:	
		
		int in_dim;
		int in_rows;
		int in_cols;
		
		int out_dim;
		int out_rows;
		int out_cols;
		
		Tensor in;
		Tensor out;
		
		virtual char getType() { }
		int getDim() { return out_dim; }
		int getRows() { return out_rows; }
		int getCols() { return out_cols; }
		
		virtual Tensor feedforward( Tensor in )  { }
		virtual Tensor feedback( Tensor delta )  { }
		virtual void updateweights( float rate, float mom ) { }
			
		Layer() { }
				
};

#endif
