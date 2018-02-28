/** 
 *  @file    Layer.h
 *  @author  Philip Salmony (pms67@cam.ac.uk)
 *  @date    10/12/2017  
 *  @version 1.0 
 *  
 *  @brief PiCNN, Layer
 *
 *  @section DESCRIPTION
 *  
 *  
 *
 */

#ifndef LAYER_H
#define LAYER_H

#include "Tensor.h"
#include "Activation.h"
#include <thread>
#include <cmath>

class Layer {
		
	public:	
		
		int in_dim;
		int in_rows;
		int in_cols;
		
		int out_dim;
		int out_rows;
		int out_cols;
		
		Tensor in;
		Tensor in_w; //Weighted inputs
		Tensor out;
		
		virtual char getType() { }
		int getDim() { return out_dim; }
		int getRows() { return out_rows; }
		int getCols() { return out_cols; }
		
		virtual void initweights( std::default_random_engine generator, double mean, double stddev ) { }
		virtual Tensor feedforward( Tensor in )  { }
		virtual Tensor feedback( Tensor delta )  { }
		virtual void updateweights( float rate, float mom ) { }
			
		Layer() { }
				
};

#endif
