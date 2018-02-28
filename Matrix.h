/** 
 *  @file    Matrix.h
 *  @author  Philip Salmony (pms67@cam.ac.uk)
 *  @date    10/12/2017  
 *  @version 1.0 
 *  
 *  @brief PiCNN, Matrix, 2D double container class
 *
 *  @section DESCRIPTION
 *  
 *  The Matrix class is used to store double values in a 2D array.
 *
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <sstream>

class Matrix {
	
	int rows, cols;
	std::vector<std::vector<double> > val;
		
	public:
		
		//Constructors
		Matrix() {
			
			rows = 0; cols = 0;
			
		}
		
		Matrix (int n) {
			
			rows = n; cols = n;
			resize();		
			
		}
		
		Matrix (int r, int c) {
			
			rows = r; cols = c;
			resize();
			
		}
			
		//Operators		
		double& operator()(int row, int col) { return val[row][col]; }
		const double& operator()(int row, int col) const { return val[row][col]; }
					
		//Properties
		int getRows() { return rows; }
		int getCols() { return cols; }
		
		//Functions
		Matrix copy() {
			
			Matrix mnew(rows, cols);
			
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					mnew(i, j) = val[i][j];
					
			return mnew;
			
		}
		
		void resize() {
		
			val.resize(rows);
			for (int i = 0; i < rows; i++)
				val[i].resize(cols);
		
			clear();
					
			return;
		
		}
		
		void resize(int rows, int cols) {
			
			this->rows = rows;
			this->cols = cols;
			resize();
			
			return;
			
		}
		
		void resize(int dim) {
			
			resize(dim, dim);
			return;
			
		}
		
		void clear() {
			
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					val[i][j] = 0;
					
			return;
			
		}
		
		void set(double n) {
			
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					val[i][j] = n;
					
			return;
			
		}
		
		void rand(float min, float max) {
			
			std::default_random_engine generator;
			std::uniform_real_distribution<double> distribution(min, max);
			
			generator.seed(time(0));
			
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					val[i][j] = distribution(generator);
					
			return;
			
		}
		
		void randn(float mean, float stddev) {
			
			std::default_random_engine generator;
			std::normal_distribution<double> distribution(mean, stddev);
			
			generator.seed(time(0));
			
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					val[i][j] = distribution(generator);
					
			return;
			
		}
		
		void print() {
			
			if (rows == 0 && cols == 0)
				return;
			
			std::stringstream s;
						
			for (int i = 0; i < rows; i ++) {
				
				s << "[ ";
			
				for (int j = 0; j < cols; j++) {
				
					s << val[i][j] << " ";
					
				}

				s << "]" << std::endl;
				
			}
			
			std::cout << s.str();
			
			return;
			
		}
	
};

#endif
