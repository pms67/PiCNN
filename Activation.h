#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>
#include <iostream>

class Activation {
	
	public:
				
		Activation() { }
		
		char type;	
	
		virtual double activate( double val ) { }
		virtual double derivative( double val ) { }	
	
};

class Sigmoid : public Activation {
	
	public:
		
		Sigmoid() { type = 's';	}
		
		double activate( double val ) {
		
			return 1 / ( 1 + exp(-val));
		
		}
		
		double derivative( double val ) {
			
			double temp = activate(val);
			
			return temp * ( 1 - temp );
			
		}
	
};

class Tanh : public Activation {
	
	public:
		
		Tanh() { type = 't'; }
		
		double activate( double val ) {
			
			return tanh(val);
		
		}
		
		double derivative( double val ) {
						
			return 1 - pow(tanh(val), 2);
			
		}
	
};

class ArcTan : public Activation {
	
	public:
		
		ArcTan() { type = 'a'; }
		
		double activate( double val ) {
		
			return atan(val);
		
		}
		
		double derivative( double val ) {
						
			return 1 / (pow(val, 2) + 1);
			
		}
	
};

class ReLU : public Activation {
	
	public:
		
		ReLU() { type = 'r'; }
		
		double activate( double val ) {
		
			if (val > 0)
				return val;
			else
				return 0;
		
		}
		
		double derivative( double val ) {
			
			if (val > 0)
				return 1;
			else
				return 0;
			
		}
	
};

class LeakyReLU : public Activation {
	
	public:
		
		LeakyReLU() { type = 'l'; }
		
		double activate( double val ) {
		
			if (val > 0)
				return val;
			else
				return 0.01 * val;
		
		}
		
		double derivative( double val ) {
			
			if (val > 0)
				return 1;
			else
				return 0.01;
			
		}
	
};

class FastSigmoid : public Activation {
	
	public:
	
		FastSigmoid() { type = 'f'; }
		
		double fast_exp( double x ) {
			
			x = 1.0 + x / 1024;
			x *= x; x *= x; x *= x; x *= x;
			x *= x; x *= x; x *= x; x *= x;
			x *= x; x *= x;
			return x;
			
		}
		
		double activate( double val ) {
		
			return 1 / (1 + fast_exp(-val));
		
		}
		
		double derivative( double val ) {
			
			double temp = activate(val);
			
			return temp * (1 - temp);
			
		}
		

	
};

#endif
