#include <iostream>
#include "Matrix.h"
#include <cmath>
#include <ctime>
#include "lConv.h"

using namespace std;

double measurespeed(int insize, int wsize, int iterations);
void numgrad(Matrix in, Matrix w, float bias, Matrix target, float step);
float getCost(Matrix out, Matrix target, Matrix & delta);
Matrix ff_conv(Matrix in, Matrix weights, float bias);
void fb_conv(Matrix in, Matrix weights, Matrix out, Matrix delta, Matrix & dCdX, Matrix & dCdW, float & dCdB);

void measureit() {
	
	int iterations = 10;
	
	cout << "Convolution speed test (Forward pass)\n";
	cout << "InSize\tKernelSize\tTime (ms)\n";
	
	for (int n = 6; n < 11; n++) {
		
		int insize = pow(2, n);
		
		for (int ksize = 3; ksize < 12; ksize += 2) {
			
			double elapsed = measurespeed(insize, ksize, iterations);
			
			cout << insize << "\t" << ksize << "\t" << elapsed << endl;
						
		}		
		
	}
	
	return;
	
}

double measurespeed(int insize, int wsize, int iterations) {
	
	Matrix in(insize);
	in.randn(1.0, 3.0);
	
	Matrix w(wsize);
	w.randn(0.0, 1.0);
	
	float bias = 0.1;
	
	clock_t begin = clock();
	
	for (int n = 0; n < iterations; n++)
		ff_conv(in, w, bias);
	
	clock_t end = clock();
	
	return (double(end - begin) / (1000 * iterations * CLOCKS_PER_SEC));
	
	
}

void checkgrad() {
	
	Matrix in(9);
	Matrix w(3);
	float bias;
	Matrix target(7);
	Matrix delta(7);
	
	in.set(2);	
	w.randn(1.0, 0.1);
	bias = 0.1;
	target.set(1);
	
	Matrix out = ff_conv(in, w, bias);
			
	cout << "Cost: " << getCost(out, target, delta) << endl;
	
	Matrix dCdX(9);
	Matrix dCdW(3);
	float dCdB;
	
	cout << "Backpropagation:" << endl;
	fb_conv(in, w, out, delta, dCdX, dCdW, dCdB);
	
	cout << "dCdB: " << dCdB << endl;
	cout << "dCdX:\n";
	dCdX.print();
	cout << "dCdW:\n";
	dCdW.print();
	
	numgrad(in, w, bias, target, 0.001);
	
	return;
	
}

void numgrad(Matrix in, Matrix w, float bias, Matrix target, float step) {
	
	cout << "Numerical Check\n";
	
	Matrix delta(in.getRows() - w.getRows() + 1, in.getCols() - w.getCols() + 1);
		
	Matrix out1 = ff_conv(in, w, bias);
	float cost1 = getCost(out1, target, delta);
	
	//dCdB
	
	float bias1 = bias + step;
	
	Matrix out2 = ff_conv(in, w, bias1);
	float cost2 = getCost(out2, target, delta);
	
	cout << "dCdB: " << (cost2 - cost1) / step << endl;
	
	
	//dCdX
	Matrix dCdX(in.getRows(), in.getCols());
		
	for (int i = 0; i < in.getRows(); i++) {
		
		for (int j = 0; j < in.getCols(); j++) {
		
			in(i, j) += step;
			
			out2 = ff_conv(in, w, bias);
			cost2 = getCost(out2, target, delta);
			
			dCdX(i, j) = (cost2 - cost1) / step;
			in(i, j) -= step;			
			
		}
				
	}
	
	cout << "dCdX:\n";
	dCdX.print();
	
	//dCdW
	Matrix dCdW(w.getRows(), w.getCols());
		
	for (int i = 0; i < w.getRows(); i++) {
		
		for (int j = 0; j < w.getCols(); j++) {
		
			w(i, j) += step;
			
			out2 = ff_conv(in, w, bias);
			cost2 = getCost(out2, target, delta);
			
			dCdW(i, j) = (cost2 - cost1) / step;
			w(i, j) -= step;			
			
		}
				
	}
	
	cout << "dCdW:\n";
	dCdW.print();
		
	return;
	
}

float getCost(Matrix out, Matrix target, Matrix & delta) {
	
	float cost = 0.0;
	float factor = out.getCols() * out.getRows();
	
	for (int i = 0; i < out.getRows(); i++)
		for (int j = 0; j < out.getCols(); j++) {

			cost += pow(out(i, j) - target(i, j), 2);
			delta(i, j) = (out(i, j) - target(i, j)) / factor;
			
		}
		
	cost /= (2 * factor);

	return cost;	
	
}

Matrix ff_conv( Matrix in, Matrix weights, float bias ) {
	
  int in_rows = in.getRows();
  int in_cols = in.getCols();	
	
  int w_size = weights.getRows();

  int out_rows = in_rows - w_size + 1;
  int out_cols = in_cols - w_size + 1;

  Matrix out(out_rows, out_cols);

  for (int m = 0; m < out_rows; m++) {

    for (int n = 0; n < out_cols; n++) {

      out(m, n) = bias; //Add bias

      for (int i = 0; i < w_size; i++) {

        for (int j = 0; j < w_size; j++) {

          //Check bounds and convolve
          if (m - i >= 0 && n - j >= 0 && m - i < in_rows && n - j < in_cols)
	        out(m, n) += in(m - i, n - j) * weights(i, j);

        }

      }

      //Apply non-linearity (ReLU)
      if (out(m, n) < 0) 
        out(m, n) = 0;

    }

  }

  return out;

}

void fb_conv( Matrix in, Matrix weights, Matrix out, Matrix delta, Matrix & dCdX, Matrix & dCdW, float & dCdB ) {
	
  int in_rows = in.getRows();
  int in_cols = in.getCols();	
	
  int w_size = weights.getRows();

  int out_rows = out.getRows();
  int out_cols = out.getCols();

  dCdB = 0;

  for (int m = 0; m < out_rows; m++) {

    for (int n = 0; n < out_cols; n++) {

      if (out(m, n) > 0) { //ReLu derivative property

        //Bias
        dCdB += delta(m, n);

        //Deltas
        for (int a = 0; a < in_rows; a++) {

          for (int b = 0; b < in_cols; b++) {

            if (m - a >= 0 && n - b >= 0 && m - a < w_size && n - b < w_size)
              dCdX(a, b) += delta(m, n) * weights(m - a, n - b);

          }

        }

        //Weights
        for (int a = 0; a < w_size; a++) {

          for (int b = 0; b < w_size; b++) {

            if (m - a >= 0 && n - b >= 0 && m - a < in_rows && n - b < in_cols)
	          dCdW(a, b) += delta(m , n) * in(m - a, n - b);

          }

        }

      }

    }

  }

  return;

}	
