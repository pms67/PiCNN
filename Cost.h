#ifndef COST_H
#define COST_H

#include "Tensor.h"
#include <cmath>

class Cost {
	
	public:
		
		Cost() { }
		
		char type;
	
		virtual double evaluate( Tensor out, Tensor target ) { }
		virtual double evaluate( Tensor out, Tensor target, Tensor & delta ) { }
	
};

class MSE : public Cost {
	
	public:
		
		MSE() { type = 'm';	}
		
		double evaluate( Tensor out, Tensor target ) {
			
			double cost = 0.0;
			
			for (int d = 0; d < out.getDim(); d++)				
				for (int r = 0; r < out.getRows(); r++)					
					for (int c = 0; c < out.getCols(); c++)						
						cost += pow(out(d, r, c) - target(d, r, c), 2);
			
			cost /= (2 * out.getDim() * out.getRows() * out.getCols());
			
			return cost;			
			
		}
		
		double evaluate( Tensor out, Tensor target, Tensor & delta ) {
			
			double cost = 0.0;
			
			for (int d = 0; d < out.getDim(); d++)				
				for (int r = 0; r < out.getRows(); r++)					
					for (int c = 0; c < out.getCols(); c++)	{
					
						delta(d, r, c) = (out(d, r, c) - target(d, r, c)) / (out.getDim() * out.getRows() * out.getCols());
						cost += pow(out(d, r, c) - target(d, r, c), 2);
			
					}
			
			cost /= (2 * out.getDim() * out.getRows() * out.getCols());
			
			return cost;			
			
		}
	
};

class CrossEntropy : public Cost {
	
	public:
		
		CrossEntropy() { type = 'c'; }
		
		double evaluate( Tensor out, Tensor target ) {
			
			double cost = 0.0;
			
			for (int d = 0; d < out.getDim(); d++)				
				for (int r = 0; r < out.getRows(); r++)					
					for (int c = 0; c < out.getCols(); c++)						
						cost += target(d, r, c) * log(out(d, r, c)) + (1 - target(d, r, c)) * log(1 - out(d, r, c));
			
			cost /= -(out.getDim() * out.getRows() * out.getCols());
			
			return cost;			
			
		}
		
		double evaluate( Tensor out, Tensor target, Tensor & delta ) {
			
			double cost = 0.0;
			
			for (int d = 0; d < out.getDim(); d++)				
				for (int r = 0; r < out.getRows(); r++)					
					for (int c = 0; c < out.getCols(); c++)	{
					
						delta(d, r, c) = -( (target(d, r, c) - out(d, r, c)) / ( out(d, r, c) * ( 1 - out(d, r, c) ) ) ) / (out.getDim() * out.getRows() * out.getCols());
						cost += target(d, r, c) * log(out(d, r, c)) + (1 - target(d, r, c)) * log(1 - out(d, r, c));
			
					}
			
			cost /= -(out.getDim() * out.getRows() * out.getCols());
			
			return cost;			
			
		}
	
};

class KLDivergence : public Cost {
	
	public:
		
		KLDivergence() { type = 'k'; }
		
		double evaluate( Tensor out, Tensor target ) {
			
			double cost = 0.0;
			
			for (int d = 0; d < out.getDim(); d++)				
				for (int r = 0; r < out.getRows(); r++)					
					for (int c = 0; c < out.getCols(); c++)						
						cost += target(d, r, c) * log(target(d, r, c) / out(d, r, c));
						
			return cost;			
			
		}
		
		double evaluate( Tensor out, Tensor target, Tensor & delta ) {
			
			double cost = 0.0;
			
			for (int d = 0; d < out.getDim(); d++)				
				for (int r = 0; r < out.getRows(); r++)					
					for (int c = 0; c < out.getCols(); c++)	{
					
						delta(d, r, c) = -target(d, r, c) / out(d, r, c);
						cost += target(d, r, c) * log(target(d, r, c) / out(d, r, c));
			
					}
			
			return cost;			
			
		}
	
};

#endif
