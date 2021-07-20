#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <assert.h>

class Mat {
private:
	int n, m;
	double **a;
public:
	Mat() {
		n = m = 0;
		a = NULL;
	}
	Mat(int n) {
		this->n = n;
		a = new double*[n];
	}
	Mat(const Mat &M) {
		n = M.rows();
		m = M.cols();
		a = new double*[n];
		for(int i=0; i<n; ++i) {
			a[i] = new double[m];
			for(int j=0; j<m; ++j)
				a[i][j] = M(i,j);
		}
	}
	Mat(int n, int m, double x=0) {
		this->n = n;
		this->m = m;
		a = new double*[n];
		for(int i=0; i<n; ++i) {
			a[i] = new double[m];
			for(int j=0; j<m; ++j)
				a[i][j] = x;
		}
	}
	int rows() const {
		return n;
	}
	int cols() const {
		return m;
	}
	void resize(int i, int j, double filler = 0) {
		if (i <= n && j <= m) {
			n = i;
			m = j;
			return;
		}
		double **tmp = new double*[i];
		for(int k=0; k<i; ++k) {
			tmp[k] = new double[j];
			for(int l=0; l<j; ++l) {
				if (l < m && k < n) tmp[k][l] = a[k][l];
				else tmp[k][l] = filler;
			}
		}
		for(int i=0; i<n; ++i)
			delete[] a[i];
		delete[] a;
		n = i;
		m = j;
		a = tmp;
	}
	void operator= (const Mat &b) {
		resize(b.rows(), b.cols());
		for(int i=0; i<n; ++i)
			for(int j=0; j<m; ++j)
				a[i][j] = b(i,j);
	}
	double& operator()(int i, int j) const {
		assert(i < n && j < m);
		return a[i][j];
	}
	Mat operator() (int k) const {
		Mat res(1, m, 0);
		for(int i=0; i<m; ++i)
			res(0,i) = a[k][i];
		return res;
	}
	void insert_row(int k, Mat v) {
		assert(v.cols() == m);
		if (k > n)
			return;
		++n;
		double **tmp = new double*[n];
		for(int i=0; i<k; ++i) {
			tmp[i] = new double[m];
			for(int j=0; j<m; ++j)
				tmp[i][j] = a[i][j];
		}
		tmp[k] = new double[m];
		for(int i=0; i<m; ++i)
			tmp[k][i] = v(0,i);
		for(int i=k+1; i<n; ++i) {
			tmp[i] = new double[m];
			for(int j=0; j<m; ++j)
				tmp[i][j] = a[i-1][j];
		}
		for(int i=0; i<n-1; ++i)
			delete[] a[i];
		delete[] a;
		a = tmp;
	}
	void remove_row(int k) {
		--n;
		for(int i=k; i<n; ++i) {
			for(int j=0; j<m; ++j)
				a[i][j] = a[i+1][j];
		}
	}
	~Mat() {
		for(int i=0; i<n; ++i)
			delete[] a[i];
		delete[] a;
	}
	friend Mat operator+ (const Mat &, const Mat &);
	friend Mat operator- (const Mat &, const Mat &);
	friend Mat inv(Mat);
	friend Mat operator* (const Mat &, const Mat &);
	friend Mat operator* (const double, const Mat &);
	friend Mat trans(const Mat &);
	friend void disp(const Mat &);
	friend void dim(const Mat &);
	friend Mat operator+(const double, const Mat &);
	friend Mat operator-(const double, const Mat &);
	friend Mat operator/(const Mat &, const double);
};

void dim(const Mat &a) {
	std::cout << a.rows() << ", " << a.cols() << "\n";
}

Mat operator/(const Mat &a, const double x) {
	Mat res(a);
	for(int i=0; i<a.n; ++i)
		for(int j=0; j<a.m; ++j)
			res(i,j) /= x;
	return res;
}

Mat operator+(const double x, const Mat &a) {
	assert(a.n == 1 && a.m == 1);
	return Mat(1, 1, x + a(0,0));
}

Mat operator-(const double x, const Mat &a) {
	assert(a.n == 1 && a.m == 1);
	return x + (-1 * a);
}

Mat operator-(const Mat &a, const Mat &b) {
	return a + ((-1) * b);
}

Mat operator* (const Mat &a, const Mat &b) {
	assert(a.m == b.n);
	Mat res(a.n, b.m, 0);
	for(int i=0; i<a.n; ++i)
		for(int j=0; j<b.m; ++j)
			for(int k=0; k<a.m; ++k)
				res(i,j) += a(i,k) * b(k,j);
	return res;
}

Mat operator+ (const Mat &a, const Mat &b) {
	assert(a.n == b.n && a.m == b.m);
	Mat res(a.n, a.m, 0);
	for(int i=0; i<a.n; ++i)
		for(int j=0; j<a.m; ++j)
			res(i,j) = a(i,j) + b(i,j);
	return res;
}

// invert Matrix using Gauss - Jordan elimination
Mat inv(Mat M) {
	assert(M.n == M.m);		// must be square matrix

	Mat inverse(M.n, M.m, 0);

	for(int i=0; i<M.n; ++i)		// identity matrix (n x n)
		inverse(i,i) = 1;

	for(int i=0; i<M.n; ++i)
	{
		double x = M(i,i);
		for(int j=M.m-1; j>=0; --j)
		{
			inverse(i,j) /= x;
			M(i,j) /= x;
		}
		for(int j=0; j<M.n; ++j)
		{
			if (i == j)
				continue;
			double x = M(j,i) * M(i,i);
			for(int k=0; k<M.m; ++k)
			{
				M(j,k) -= x * M(i,k);
				inverse(j,k) -= x * inverse(i,k);
			}
		}
	}
	return inverse;
}

void disp(const Mat &M) {
	std::cout << "Matrix :=\n\n";
	for(int i=0; i<M.n; ++i) {
		for(int j=0; j<M.m; ++j)
			std::cout << M(i,j) << " ";
		std::cout << "\n";
	}
	std::cout << "\n";
}

Mat trans(const Mat &a) {
	Mat res(a.m,a.n);
	for(int i=0; i<a.n; ++i)
		for(int j=0; j<a.m; ++j)
			res(j,i) = a(i,j);
	return res;
}


Mat operator* (const double x, const Mat &a) {
	Mat res(a);
	for(int i=0; i<a.n; ++i)
		for(int j=0; j<a.m; ++j)
			res(i,j) *= x;
	return res;
}

#endif