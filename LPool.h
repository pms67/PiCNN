#ifndef POOLLAYER_H
#define POOLLAYER_H

#include "Layer.h"
#include <cmath>

class LPool : public Layer {
	
	private:
		
		//Properties
		int p_size;
		
		//Activation matrix
		Tensor activation;
			
		//Gradients
		Tensor dCdX;
		Tensor delta;
		
		//Threading
		std::vector<bool > thread_status;
		std::vector<bool > thread_status_fb;
		
	public:
				
		//Constructor
		LPool( int in_dim, int in_rows, int in_cols, int p_size ) {
			
			//Set dimensions
			this->in_dim = in_dim;
			this->out_dim = in_dim;
			this->in_rows = in_rows;
			this->in_cols = in_cols;
			this->p_size = p_size;
			
			out_rows = floor(in_rows / p_size);
			out_cols = floor(in_cols / p_size);
			
			//Redimension tensors
			in.resize(in_dim, in_rows, in_cols);
			out.resize(in_dim, out_rows, out_cols);
			activation.resize(in_dim, in_rows, in_cols);
			delta.resize(in_dim, out_rows, out_cols);
			dCdX.resize(in_dim, in_rows, in_cols);
			
			//Threading
			thread_status.resize(in_dim);
			thread_status_fb.resize(in_dim);
										
		}
		
		//Properties
		char getType() { return 'p'; }
		
		//Functions
		//Threaded feedforward version, d specifies which dimension/feature-map is computed
		void feedforward_dim( int d ) {
			
			thread_status[d] = true;
			
			int outx = 0;
		
			for (int m = 0; m < in_rows - p_size + 1; m += p_size) {
			
				int outy = 0;
			
				for (int n = 0; n < in_cols - p_size +1; n += p_size) {
				
					double max = -99999.0; int maxx = 0; int maxy = 0;
				
					for (int i = 0; i < p_size; i++) {
					
						for (int j = 0; j < p_size; j++) {
						
							if (in(d, m + i, n + j) > max) {
						
								max = in(d, m + i, n + j);
								maxx = m + i;
								maxy = n + j;
							
							}
						
						}
					
					}
				
					out(d, outx, outy) = max;
					activation(d, maxx, maxy) = 1;
				
					outy += 1;				
				
				}
			
				outx += 1;
			
			}
			
			thread_status[d] = false;			
			return;
			
		}
				
		Tensor feedforward( Tensor in ) {
						
			this->in = in.copy();
			activation.set(0);
			
			//Threading is only computationaly beneficial for larger dimensions
			if (in_dim > 1 && (in_rows * in_cols > 1024)) {
				
				//Create and start threads
				std::thread t[in_dim];
		
				for (int d = 0; d < in_dim; d++)	
					t[d] = std::thread([=] { feedforward_dim(d); });
				
				//Check if all threads are completed
				bool active = true;
			
				while (active) {
				
					active = false;
				
					for (int d = 0; d < in_dim; d++)
						if (thread_status[d])
							active = true;
										
				}
		  					
				//Join threads to main thread
				for (int d = 0; d < in_dim; d++)
					t[d].join();
				
			} else {
						
				for (int d = 0; d < in_dim; d++) {
							
					int outx = 0;
				
					for (int m = 0; m < in_rows - p_size + 1; m += p_size) {
					
						int outy = 0;
					
						for (int n = 0; n < in_cols - p_size +1; n += p_size) {
						
							double max = -99999.0; int maxx = 0; int maxy = 0;
						
							for (int i = 0; i < p_size; i++) {
							
								for (int j = 0; j < p_size; j++) {
								
									if (in(d, m + i, n + j) > max) {
								
										max = in(d, m + i, n + j);
										maxx = m + i;
										maxy = n + j;
									
									}
								
								}
							
							}
						
							out(d, outx, outy) = max;
							activation(d, maxx, maxy) = 1;
						
							outy += 1;				
						
						}
					
						outx += 1;
					
					}
				
				}
				
			}
						
			return out;			
				
		}
		
		void feedback_dim( int d ) {
			
			thread_status_fb[d] = true;
			
			int outx = 0;
		
			for (int m = 0; m < in_rows - p_size + 1; m += p_size) {
			
				int outy = 0;
			
				for (int n = 0; n < in_cols - p_size + 1; n += p_size) {
				
					for (int i = 0; i < p_size; i++) {
					
						for (int j = 0; j < p_size; j++) {
						
							if (abs(activation(d, m + i, n + j) - 1) < 1E-3)
								dCdX(d, m + i, n +j) = delta(d, outx, outy);
						
						}
					
					}
				
					outy += 1;
				
				}
			
				outx += 1;
			
			}
		
			thread_status_fb[d] = false;
			return;
			
		}
		
		Tensor feedback( Tensor delta ) {

			this->delta = delta.copy();
			dCdX.set(0);
			
			 //Threading is only computationaly beneficial for larger dimensions
			if (out_dim > 1 && (out_rows * out_cols >= 1024 )) {
  			  			
				//Create and start threads
				std::thread t_fb[in_dim];
		
				for (int d = 0; d < in_dim; d++)	
					t_fb[d] = std::thread([=] { feedback_dim(d); });
				
				//Check if all threads are completed
				bool active = true;
							
				while (active) {
				
					active = false;
				
					for (int d = 0; d < in_dim; d++)						
						if (thread_status_fb[d])
							active = true;
										
				}
						  							  					
				//Join threads to main thread			
				for (int d = 0; d < in_dim; d++)				
					t_fb[d].join();
					
			} else {
					
				for (int d = 0; d < in_dim; d++) {
							
					int outx = 0;
				
					for (int m = 0; m < in_rows - p_size + 1; m += p_size) {
					
						int outy = 0;
					
						for (int n = 0; n < in_cols - p_size + 1; n += p_size) {
						
							for (int i = 0; i < p_size; i++) {
							
								for (int j = 0; j < p_size; j++) {
								
									if (abs(activation(d, m + i, n + j) - 1) < 1E-3)
										dCdX(d, m + i, n +j) = delta(d, outx, outy);
								
								}
							
							}
						
							outy += 1;
						
						}
					
						outx += 1;
					
					}
				
				}
			
			}
			
			return dCdX;
		
		}
		
		void updateweights( float rate, float mom ) { return; }	
	
};


#endif
