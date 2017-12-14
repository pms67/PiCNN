#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>

double act_relu( double val ) {
	
	if (val > 0)
		return val;
	else
		return 0;
	
}

double act_d_relu( double val ) {
	
	if (val > 0)
		return 1;
	else
		return 0;
	
}

double act_relu_leaky( double val ) {
	
	if (val > 0)
		return val;
	else
		return 0.01 * val;
	
}

double act_d_relu_leaky( double val) {
	
	if (val > 0)
		return 1;
	else
		return 0.01;
	
}

double act_sigmoid( double val ) {
	
	return 1 / (1 + exp(-val));
		
}

double act_d_sigmoid( double val ) {
	
	double temp = act_sigmoid(val);
	
	return temp * (1 - temp);
	
}

double act_tanh( double val ) {
	
	return (2 / (1 + exp(-2*val))) - 1;
	
}

double act_d_tanh( double val ) {
	
	double temp = pow(act_tanh(val), 2);
	
	return 1 - temp;
	
}

#endif
