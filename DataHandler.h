#ifndef DATAHANDLER_H
#define DATAHANDLER_H

#include "Tensor.h"
#include <fstream>
#include <string>

class DataHandler {
	
	public:
				
		//Constructor		
		DataHandler() { 
			//Empty
		}
		
		//Functions
		TensorArray readCSV(std::string filename, std::string delimiter) {
			
			std::ifstream file(filename);
			
			if (file.is_open()) {
				
				std::vector<std::vector<std::string > > data;
				int num_rows = 0; int num_cols = 0;
				std::string current_row;
				
				while ( getline( file, current_row ) )			
					data.push_back(delimit(current_row, delimiter));
					
				num_rows = data.size();		
				num_cols = data[0].size();
				
				TensorArray out;
				out.resize(num_rows);
				
				for (int i = 0; i < num_rows; i++) {
					
					out[i].resize(1, num_cols, 1);
					
					for (int j = 0; j < num_cols; j++) {
						
						out[i](0, j, 0) = std::stod(data[i][j]);
											
					}
									
				}
				
				return out;
				
								
			} else {
				
				std::cout << "Unable to open file.\n";
				TensorArray out;				
				
				return out;
				
			}
			
		}
		
		TensorArray reshape(TensorArray data, int rows, int cols) {
			
			if (rows * cols != data[0].getRows()) {
				
				std::cout << "Rows (" << rows << ") * Cols (" << cols << ") must equal size of input tensors (" << data[0].getRows() << ")!";
				return data;
				
			}
			
			TensorArray out;
			out.resize(data.size());		
			
			//Fill tensors						
			for (int i = 0; i < data.size(); i++) {
				
				//Reshape tensors
				out[i].resize(1, rows, cols);
					
				int count = 0;
						
				for (int j = 0; j < rows; j++) {
												
					for (int k = 0; k < cols; k++) {
				
						out[i](0, j, k) = data[i](0, count, 0);
						count++;
						
					}
					
				}
				
			}
					
			return out;
			
		}
		
		TensorArray onehot(TensorArray data, int max) {
			
			if (data[0].getRows() > 1 || data[0].getCols() > 1) {
				
				std::cout << "Tensors have to be dimensions of 1x1x1.\n";
				return data;
				
			}
			
			TensorArray out;
			out.resize(data.size());
			
			for (int i = 0; i < data.size(); i++) {
			
				out[i].resize(1, max, 1);
				int hot = int(data[i](0, 0, 0));
				out[i](0, hot, 0) = 1;
				
			}
						
			return out;			
			
		}
		
		TensorArray normalise_minmax(TensorArray data, double min, double max) {
			
			if (min >= max) {
				
				std::cout << "MIN must be less than MAX.\n";
				return data;
				
			}
			
			TensorArray out;
			out.resize(data.size());
			
			for (int n = 0; n < data.size(); n++) {
				
				out[n].resize(data[0].getDim(), data[0].getRows(), data[0].getCols());
				
				for (int i = 0; i < data[0].getDim(); i++)
					for (int j = 0; j < data[0].getRows(); j++)
						for (int k = 0; k < data[0].getCols(); k++)
							out[n](i, j, k) = (data[n](i, j, k) - min) / (max - min);
				
			}
			
			return out;
			
		}
				
		std::vector<std::string > delimit(std::string s, std::string delimiter) {
						
			std::vector<std::string > out;					

			int pos = 0;

			while (pos >= 0) {
				
				pos = s.find(delimiter);
								
				std::string val = s.substr(0, pos);
						
				s.erase(0, pos + delimiter.length());
				
				out.push_back(val);
								
			}
						
			return out;
			
		}
		
		bool compare_onehot(Tensor a, Tensor b) {
			
			if (a.getCols() > 1 || b.getCols() > 1 || a.getDim() > 1 || b.getDim() > 1 || a.getRows() != b.getRows()) {
				
				std::cout << "Must be one-hot vectors (i.e. a 1xNx1 vectors) of equal length.\n";
				return false;
				
			}
			
			double maxA = -9999.0;
			int valA = 0;
			
			double maxB = -9999.0;
			int valB = 0;
			
			for (int i = 0; i < a.getRows(); i++) {
				
				if (a(0, i, 0) > maxA) {
					
					maxA = a(0, i, 0);
					valA = i;
					
				}
				
				if (b(0, i, 0) > maxB) {
					
					maxB = b(0, i, 0);
					valB = i;
					
				}
								
			}
			
			if (valA == valB)
				return true;
			else
				return false;
			
		}
		
		double getLearningRate(float val_acc, float lmin, float lmax, float amin, float amax) {
			
			double m = (lmax - lmin) / (amin - amax);
			double c = lmin - m * amax;
			
			double lrate = (m * val_acc + c);			
			
			if (lrate > lmax)
				lrate = lmax;
			else if (lrate < lmin)
				lrate = lmin;
			
			return lrate;
			
		}
			
};

#endif
