#include "Matrix.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>
using namespace std;

// Hand Written Digit Classification
const int MAX_ITER = 25;
const int MAX_INPUTS = 8000;
const int BATCH_SIZE = 800;
double LR = 1;
const double LAMBDA = 0.5;
const int FEATURES = 784;
const int LAYERS = 3;
const int CLASSES = 10;
int network[] = {FEATURES, 35, CLASSES};
Mat X(2 * MAX_INPUTS, FEATURES, 0), Y(2 * MAX_INPUTS, 1, 0);

Mat sig(Mat x) {
	for(int i=0; i<x.rows(); ++i)
		for(int j=0; j<x.cols(); ++j)
			x(i,j) = double(1) / (1 + exp(-x(i,j)));
	return x;
}


Mat Log(Mat x) {
	for(int i=0; i<x.rows(); ++i)
		for(int j=0; j<x.cols(); ++j)
			x(i,j) = log10(x(i,j));
	return x;
}

Mat* feedForward(int idx, Mat *weights) {
	Mat *a = new Mat[LAYERS], *z = new Mat[LAYERS];
	z[0] = trans(X(idx));
	a[0] = z[0];
	a[0].insert_row(0, Mat(1,1,1));
	for(int j=1; j<LAYERS; ++j) {
		z[j] = weights[j-1] * a[j-1];
		a[j] = sig(z[j]);
		if (j < LAYERS - 1) a[j].insert_row(0, Mat(1,1,1));
	}
	delete[] z;
	return a;
}

int predict(int idx, Mat *weights) {
	Mat *a = feedForward(idx, weights);
	double prediction = -1;
	int digit;
	for(int i=0; i<CLASSES; ++i) {
		if (prediction < a[LAYERS-1](i,0)) {
			prediction = a[LAYERS-1](i,0);
			digit = i;
		}
	}
	delete[] a;
	return digit;
}

Mat* computeGradients(int start, int end, Mat *weights, Mat *grads) {
	int m = (end - start);
	int n = X.cols();
	Mat er(1,1,0);
	for(int i=0; i<LAYERS-1; ++i)
		grads[i] = Mat(network[i+1], network[i]+1, 0);
	Mat *a = new Mat[LAYERS],  *z = new Mat[LAYERS];
	Mat *S = new Mat[LAYERS];
	for(int i=start; i<end; ++i) {
		// forward propogation
		z[0] = trans(X(i));
		a[0] = z[0];
		a[0].insert_row(0, Mat(1,1,1));
		for(int j=1; j<LAYERS; ++j) {
			z[j] = weights[j-1] * a[j-1];
			a[j] = sig(z[j]);
			if (j < LAYERS - 1) a[j].insert_row(0, Mat(1,1,1));
		}

		// compute error
		Mat y_aug(CLASSES, 1);
		for(int j=0; j<CLASSES; ++j)
				y_aug(j,0) = (Y(i,0) == j);
		er = er + (trans(y_aug) * Log(a[LAYERS-1]));
		er = er + (trans(Mat(CLASSES,1,1) - y_aug) * Log(Mat(a[LAYERS-1].rows(),1,1) - a[LAYERS-1]));
		

		// back propogation
		S[LAYERS-1] = (a[LAYERS-1] - y_aug);
		for(int j=LAYERS-2; j; --j) {
			S[j] = trans(weights[j]) * S[j+1];
			for(int k=0; k<S[j].rows(); ++k)
				S[j](k,0) = S[j](k,0) * a[j](k,0) * (1 - a[j](k,0));
			S[j].remove_row(0);
			grads[j] = grads[j] + S[j+1] * trans(a[j]);
		}
		grads[0] = grads[0] + S[1] * trans(a[0]);
	}
	double reg = 0;
	for(int i=0; i<LAYERS-1; ++i) {
		grads[i] = grads[i] / m;
		for(int j=0; j<network[i+1]; ++j) {
			for(int k=1; k<=network[i]; ++k) {
				grads[i](j,k) += (LAMBDA / m) * weights[i](j,k);
				reg += weights[i](j,k) * weights[i](j,k);
			}
		}
	}
	delete[] a;
	delete[] S;
	delete[] z;
	reg = reg * (LAMBDA / (2*m));
	er = (-1 * er) / m;
	cout << er(0,0) + reg;
	return grads;
}

void batchgradientDescent(Mat *weights) {
	Mat *gradients = new Mat[LAYERS-1];
	int end;
	cout << "\tIteration\tError\n\n";
	int Iter = MAX_ITER;
	while (Iter > 0) {
		cout << "\t" << Iter << "\t\t";
		for(int cur_batch = 0; cur_batch < MAX_INPUTS; cur_batch += BATCH_SIZE) {
			end = min(MAX_INPUTS, cur_batch + BATCH_SIZE);
			computeGradients(cur_batch, end, weights, gradients);
			for(int i=0; i<LAYERS-1; ++i)
				weights[i] = weights[i] - LR * gradients[i];
			cout << ", ";
		}
		cout << "\n";
		--Iter;
	}
	delete[] gradients;
}

void randomInit(Mat *weights) {
	for(int l=0; l<LAYERS-1; ++l) {
		double e_init = sqrt(double(6) / (network[l] + network[l+1]));
		int range = (network[l] + 1) * network[l+1];
		for(int i=0; i<weights[l].rows(); ++i) {
			for(int j=0; j<weights[l].cols(); ++j) {
				weights[l](i,j) =  e_init * (double(rand() % (range*2+1)) / range) - e_init;
			}
		}
	}
}

uint32_t convert(int n) {
	int byte[4];
	int i = 0;
	int mul = 1 << 24, res = 0;
	while (i < 4) {
		res += (n & 255) * mul;
		mul >>= 8;
		n >>= 8;
		i++;
	}
	return res;
}

void inputImages() {
	ifstream in("train-images.idx3-ubyte", ios::binary);
	ifstream out("train-labels.idx1-ubyte", ios::binary);
	int magic;
	out.read((char*)&magic, sizeof(magic));
	in.read((char*)&magic, sizeof(magic));
	int labels;
	out.read((char *)&labels, sizeof(labels));
	int images;
	in.read((char *)&images, sizeof(images));
	labels = convert(labels);
	images = convert(images);
	images = min(images, 2 * MAX_INPUTS);
	labels = min(labels, 2 * MAX_INPUTS);
	int rows;
	in.read((char *)&rows, sizeof(rows));
	rows = convert(rows);
	int cols;
	in.read((char *)&cols, sizeof(cols));
	cols = convert(cols);
	for(int i=0; i<images; ++i) {
		unsigned char byte = 0;
		out.read((char *)&byte, sizeof(byte));
		int num = int(byte);
		Y(i,0) = num;
		for(int j=0; j<rows * cols; ++j) {
			in.read((char *)&byte, sizeof(byte));
			X(i,j) = double(byte) / 255;
		}
	}
	in.close();
	out.close();
}

int main() {
	srand(time(NULL));
	inputImages();
	Mat *weights = new Mat[LAYERS - 1];
	for(int i=0; i<LAYERS-1; ++i) {
		weights[i].resize(network[i+1], network[i]+1);
	}
	randomInit(weights);
	batchgradientDescent(weights);
	cout << "\tCorrect\tGuess\n\n";
	for(int i=MAX_INPUTS; i<MAX_INPUTS + 100; ++i) {
		cout << "\t" << Y(i,0) << "\t\t";
		cout << predict(i, weights) << "\n";
	}
	return 0;
}