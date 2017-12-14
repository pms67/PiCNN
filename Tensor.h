/** 
 *  @file    Tensor.h
 *  @author  Philip Salmony (pms67@cam.ac.uk)
 *  @date    12/12/2017  
 *  @version 1.0 
 *  
 *  @brief PiCNN, Tensor, 3D double container class
 *
 *  @section DESCRIPTION
 *  
 *  The Tensor class is used to store double values in a 3D array.
 *  If the dim argument is 1, a tensor reduces to a 2D matrix. 
 *
 */

#ifndef TENSOR_H
#define TENSOR_H

#include "Matrix.h"

class Tensor {
	
	private:
		
		int dim;
		int rows;
		int cols;
		
		std::vector<Matrix > m;
		
		void redim() {
			
			m.resize(dim);
			
			for (int d = 0; d < dim; d++)
				m[d].resize(rows, cols);
			
			return;
			
		}
		
	public:
		
		//Constructors
		Tensor() {
			
			dim = 0; rows = 0; cols = 0;
			
		}
		
		Tensor(int dim, int rows, int cols) {
			
			this->dim = dim; this->rows = rows; this->cols = cols;
			redim();
			
		}
		
		Tensor(int dim, int n) {
			
			this->dim = dim; this->rows = n; this->cols = n;
			redim();
			
		}
		
		Tensor(int n) {
			
			this->dim = n; this->rows = n; this->cols = n;
			redim();
			
		}
				
		//Operators
		Matrix operator()(int d) { return m[d]; }
		const Matrix operator()(int d) const { return m[d]; }
		
		double& operator()(int d, int r, int c) { return m[d](r, c); }
		const double& operator()(int d, int r, int c) const { return m[d](r, c); }
	
		//Properties
		int getDim() { return dim; }
		int getRows() { return rows; }
		int getCols() { return cols; }
	
		//Functions
		void resize(int dim, int rows, int cols) {
			
			this->dim = dim; this->rows = rows; this->cols = cols;
			redim();
			
		}
		
		void resize(int dim, int n) {
			
			this->dim = dim; this->rows = n; this->cols = n;
			redim();
			
		}
		
		void resize(int n) {
			
			this->dim = n; this->rows = n; this->cols = n;
			redim();
			
		}
		
		Tensor copy() {
			
			Tensor tnew(dim, rows, cols);
			
			for (int d = 0; d < dim; d++)
				for (int i = 0; i < rows; i++)
					for (int j = 0; j < cols; j++)
						tnew(d, i, j) = m[d](i, j);
					
			return tnew;
			
		}
		
		void clear() {
			
			for (int d = 0; d < dim; d++)
				for (int i = 0; i < rows; i++)
					for (int j = 0; j < cols; j++)
						m[d](i, j) = 0;
					
			return;
			
		}
	
		void set(int d, Matrix mset) {
			
			m[d] = mset.copy();			
			return;
			
		}
	
		void set(int d, double val) {
			
			m[d].set(val);
			return;
			
		}
	
		void set(double val) {
						
			for (int d = 0; d < dim; d++)
				m[d].set(val);
					
			return;
			
		}
		
		void rand(float min, float max) {
						
			std::default_random_engine generator;
			std::uniform_real_distribution<double> distribution(min, max);
			
			generator.seed(time(0));
			
			for (int d = 0; d < dim; d++)
				for (int i = 0; i < rows; i++)
					for (int j = 0; j < cols; j++)
						m[d](i, j) = distribution(generator);
					
			return;
			
		}
		
		void randn(float mean, float stddev) {
			
			std::default_random_engine generator;
			std::normal_distribution<double> distribution(mean, stddev);
			
			generator.seed(time(0));
			
			for (int d = 0; d < dim; d++)
				for (int i = 0; i < rows; i++)
					for (int j = 0; j < cols; j++)
						m[d](i, j) = distribution(generator);
					
			return;
			
		}
		
		void print() {
			
			for (int d = 0; d < dim; d++) {
			
				std::cout << d << ":\n"; 
				m[d].print();
			
			}
			
			return;
			
		}

		void print(int n) {
						
			m[n].print();
			return;
			
		}
	
};

typedef std::vector<Tensor > TensorArray;

#endif
