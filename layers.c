#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "utilities.h"
#include "layers.h"

//		Y = AX + b  			&Y,      A,   		X,		B,     Rows (for A), Columns (for A)
void fully_connected_forward(double* Y, double* A, double* X, double* b, int R, int C)
{
	for (int i = 0; i < R; i++) {
		Y[i] = b[i];
		for (int n = 0; n < C; n++) {
			Y[i] += A[i * C + n] * X[n];
		}
	}

}
//		Y = AX + b  			dldY,       A,     X,        &dldA,    &dldX,    &dldb   Rows (A), Columns (A)
void fully_connected_backward(double* dldY, double* A, double* X, double* dldA, double* dldX, double* dldb, int R, int C)
{
	// computing dldA
	for (int i = 0; i < R; i++) {
		for (int n = 0; n < C; n++) {
			dldA[i * C + n] = dldY[i] * X[n];
		}
	}

	// computing dldb
	for (int i = 0; i < R; i++) {
		dldb[i] = dldY[i];
	}

	// computing dldX 
	for (int i = 0; i < C; i++) {
		dldX[i] = 0.0;
		for (int n = 0; n < R; n++) {
			dldX[i] += A[n * C + i] * dldY[n];
		}
	}
}

double mean_square_error(double* probs, int F, int correct) {
	double mse = 0;
	for (int i = 0; i < F; i++) {
		if (i == correct)
			mse += pow(probs[i] - 1.0, 2.0);
		else
			mse += pow(probs[i],2.0);
	}
	mse /= (double)F;

	return mse;
}
double cross_entropy(double* probs, int correct)
{
	if (probs[correct] <= DBL_EPSILON) {//log(0)=nan
		return -log(DBL_EPSILON);
	}
	return -log(probs[correct]);
}

// Dealing with softmax layer, forward and backward
//								&P,		Y,  	features
void softmax_layers_forward(double* P, double* Y, int F, double temperature)
{
	double sum = 0;
	//double cache[F];
	double *cache;
	init_zero_vector(&cache, F);
	//double* cache = (double*)calloc(F ,sizeof(double));
	
	for (int f = 0; f < F; f++) {
		cache[f] = exp(Y[f] / temperature);
		sum += cache[f];
	}

	for (int f = 0; f < F; f++) {
		P[f] = cache[f] / sum;
	}

	free_vector(&cache);
}
//									  P,	  c,  &dldh, rows
void softmax_loss_layer_backward(double* P, int c, double* dldh, int R)
{
	for (int r = 0; r < R; r++) {
		dldh[r] = P[r];
	}

	dldh[c] -= 1.0;
}
// Other layers used: sigmoid and tanh
// 	
// 		Y = sigmoid(X), &Y, X, length
void sigmoid_forward(double* Y, double* X, int L)
{
	for (int l = 0; l < L; l++) {
		Y[l] = 1.0 / (1.0 + exp(-X[l]));
	}
}
// 		Y = sigmoid(X), dldY, Y, &dldX, length
void sigmoid_backward(double* dldY, double* Y, double* dldX, int L)
{
	for (int l = 0; l < L; l++) {
		dldX[l] = (1.0 - Y[l]) * Y[l] * dldY[l];
	}

}
// 		Y = tanh(X), &Y, X, length
void tanh_forward(double* Y, double* X, int L)
{
	for (int l = 0; l < L; l++) {
		Y[l] = tanh(X[l]);
	}
}
// 		Y = tanh(X), dldY, Y, &dldX, length
void tanh_backward(double* dldY, double* Y, double* dldX, int L)
{
	for (int l = 0; l < L; l++) {
		dldX[l] = (1.0 - Y[l] * Y[l]) * dldY[l];
	}
}
