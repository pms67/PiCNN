#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "Layer.h"

class LConv : public Layer {
		
	private:
		
		//Properties
		int w_size;
		
		//Weights and bias
		Tensor weights;
		Tensor bias;
		
		//Gradients
		Tensor delta;
		Tensor dCdX;
		Tensor dCdW;
		Tensor dCdB;
		
		Tensor prev_dCdW;
		Tensor prev_dCdB;
		
		//Threading
		std::vector<bool > thread_status;
		std::vector<bool > thread_status_fb; //Feedback threads
		
		//Activation
		Activation* act;
		
	public:
									
		//Constructor
		LConv(int in_dim, int in_rows, int in_cols, int w_size, char act_func) {
			
			//Set dimensions
			this->in_dim = in_dim;
			this->out_dim = in_dim;
			this->in_rows = in_rows;
			this->in_cols = in_cols;
			this->w_size = w_size;
			
			out_rows = in_rows - w_size + 1;
			out_cols = in_cols - w_size + 1;
			
			//Redimension tensors
			in.resize(in_dim, in_rows, in_cols);
			in_w.resize(in_dim, in_rows, in_cols);
			out.resize(in_dim, out_rows, out_cols);
						
			weights.resize(in_dim, w_size);
			bias.resize(in_dim, 1, 1);
			
			delta.resize(in_dim, out_rows, out_cols);
			dCdX.resize(in_dim, in_rows, in_cols);
			dCdW.resize(in_dim, w_size);
			dCdB.resize(in_dim, 1, 1);
			prev_dCdW.resize(in_dim, w_size);
			prev_dCdB.resize(in_dim, 1, 1);
			
			//Threading
			thread_status.resize(in_dim);
			thread_status_fb.resize(in_dim);
			
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
		char getType() { return 'c'; }
		void print() { weights.print();	bias.print(); return; }
		void printGrads() { dCdW.print(); dCdB.print(); return;	}
				
		//Functions
		//Initialise weights randomly according to a normal distribution
		void initweights( std::default_random_engine generator, double mean, double stddev ) {
			
			std::normal_distribution<double> distribution(mean, stddev);
			
			for (int d = 0; d < in_dim; d++) {
				
				bias(d, 0, 0) = distribution(generator);
				
				for (int i = 0; i < w_size; i++)
					for (int j = 0; j < w_size; j++)
						weights(d, i, j) = distribution(generator);
			
			}
			
			return;
			
		}		
		
		//Threaded feedforward version, d specifies which dimension/feature-map is computed
		void feedforward_dim( int d ) {
			
			thread_status[d] = true;
			
			for (int m = 0; m < out_rows; m++) {
		
	    		for (int n = 0; n < out_cols; n++) {
	
	    			in_w(d, m, n) = bias(d, 0 , 0); //Add bias
	
	    			for (int i = 0; i < w_size; i++) {
	
	        			for (int j = 0; j < w_size; j++) {
	
	          				//Check bounds and convolve
	          				if (m - i >= 0 && n - j >= 0 && m - i < in_rows && n - j < in_cols)
		        				in_w(d, m, n) += in(d, m - i, n - j) * weights(d, i, j);
	
	        			}
	
					}
	
	      			//Apply non-linearity
					out(d, m, n) = act->activate(in_w(d, m, n));      			       			
	
	    		}
	
	  		}
	  					  
			thread_status[d] = false;			
			return;
			
		}
				
		Tensor feedforward( Tensor in ) {
			
			//Copy inputs			
			this->in = in.copy();
		
			//Threading is only computationaly beneficial for larger dimensions
			if (in_dim > 1 && (in_rows * in_cols >= 1024 )) {
								
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
				
					for (int m = 0; m < out_rows; m++) {
			
			    		for (int n = 0; n < out_cols; n++) {
			
			    			in_w(d, m, n) = bias(d, 0, 0); //Add bias
			
			    			for (int i = 0; i < w_size; i++) {
			
			        			for (int j = 0; j < w_size; j++) {
			
			          				//Check bounds and convolve
			          				if (m - i >= 0 && n - j >= 0 && m - i < in_rows && n - j < in_cols)
				        				in_w(d, m, n) += in(d, m - i, n - j) * weights(d, i, j);
			
			        			}
			
							}
			
			      			//Apply non-linearity (ReLU)
			      			out(d, m, n) = act->activate(in_w(d, m, n));    			       			
			
			    		}
			
			  		}
					
				}
				
			}	
				
		  	return out;
		
		}
		
		void feedback_dim( int d )  {	
			
			thread_status_fb[d] = true;
								
			for (int m = 0; m < out_rows; m++) {
			
				for (int n = 0; n < out_cols; n++) {

  					if (out(d, m, n) > 0) { //ReLu derivative property

    					//Bias
    					dCdB(d, 0, 0) += delta(d, m, n) * act->derivative(in_w(d, m, n));

    					//Deltas
    					for (int a = 0; a < in_rows; a++) {

      						for (int b = 0; b < in_cols; b++) {

        						if (m - a >= 0 && n - b >= 0 && m - a < w_size && n - b < w_size)
          							dCdX(d, a, b) += delta(d, m, n) * act->derivative(in_w(d, m, n)) * weights(d, m - a, n - b);

      						}

    					}

    					//Weights
    					for (int a = 0; a < w_size; a++) {

      						for (int b = 0; b < w_size; b++) {

        						if (m - a >= 0 && n - b >= 0 && m - a < in_rows && n - b < in_cols)
          							dCdW(d, a, b) += delta(d, m , n) *  act->derivative(in_w(d, m, n)) * in(d, m - a, n - b);

      						}

    					}

  					}

				}

			}
					
			thread_status_fb[d] = false;
			return;
			
		}
		
		Tensor feedback( Tensor delta ) {
			
			//Copy deltas
			this->delta = delta.copy();
			
			//Calculate changes in gradients
			prev_dCdW = dCdW.copy();
			prev_dCdB = dCdB.copy();			
			
			//Reset gradients
			dCdX.clear();
			dCdW.clear();
  			dCdB.clear();
  			
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
				
				for (int d = 0; d < out_dim; d++) {
								
					for (int m = 0; m < out_rows; m++) {
	
						for (int n = 0; n < out_cols; n++) {
		
		  					if (out(d, m, n) > 0) { //ReLu derivative property
		
		    					//Bias
		    					dCdB(d, 0, 0) += delta(d, m, n) * act->derivative(in_w(d, m, n));
		
		    					//Deltas
		    					for (int a = 0; a < in_rows; a++) {
		
		      						for (int b = 0; b < in_cols; b++) {
		
		        						if (m - a >= 0 && n - b >= 0 && m - a < w_size && n - b < w_size)
		          							dCdX(d, a, b) += delta(d, m, n) * act->derivative(in_w(d, m, n)) * weights(d, m - a, n - b);
		
		      						}
		
		    					}
		
		    					//Weights
		    					for (int a = 0; a < w_size; a++) {
		
		      						for (int b = 0; b < w_size; b++) {
		
		        						if (m - a >= 0 && n - b >= 0 && m - a < in_rows && n - b < in_cols)
		          							dCdW(d, a, b) += delta(d, m , n) * act->derivative(in_w(d, m, n)) * in(d, m - a, n - b);
		
		      						}
		
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
