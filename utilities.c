#include "utilities.h"

void vectors_add(double* A, double* B, int L)
{
	for (int i = 0; i < L; i++) {
		A[i] += B[i];
	}
}

void vectors_add_scalar(double* A, double B, int L)
{
	for (int i = 0; i < L; i++) {
		A[i] += B;
	}
}

void vectors_scalar_multiply(double* A, double d, int L)
{
	for (int i = 0; i < L; i++) {
		A[i] *= d;
	}
}

// A = A + (B * s)
void vectors_add_scalar_multiply(double* A, double* B, int L, double s)
{
	for (int i = 0; i < L; i++) {
		A[i] += B[i] * s;
	}
}

void vectors_substract(double* A, double* B, int L)
{
	for (int i = 0; i < L; i++) {
		A[i] -= B[i];
	}
}

void vectors_div(double* A, double* B, int L)
{
	for(int i=0;i<L;i++){
		A[i] /= B[i];
	}
}

void vector_sqrt(double* A, int L)
{
	for(int i=0;i<L;i++){
		A[i] = sqrt(A[i]);
	}
}
// A = A - (B * s)
void vectors_substract_scalar_multiply(double* A, double* B, int L, double s)
{
	for (int i = 0; i<L; i++) {
		A[i] -= B[i] * s;
	}
}


void vectors_multiply(double* A, double* B, int L)
{
	for (int i = 0; i<L; i++) {
		A[i] *= B[i];
	}
}
void vectors_mutliply_scalar(double* A, double b, int L)
{
	for (int i = 0; i<L; i++) {
		A[i] *= b;
	}
}

int init_random_matrix(double*** A, int R, int C)
{
	*A = calloc(R, sizeof(double*));

	if (*A == NULL)
		return -1;

	for(int r=0;r<R;r++) {
		(*A)[r] = calloc(C, sizeof(double));
		if ((*A)[r] == NULL)
			return -2;
	}

	for(int r=0;r<R;r++) {
		for(int c=0;c<C;c++) {
			(*A)[r][c] = randn(0, 1) / sqrt(R);
		}
	}

	return 0;
}

double*	get_random_vector(int L, int R) {
	double *p;
	p = calloc(L, sizeof(double));
	if (p == NULL)
		exit(0);

	for(int l=0;l<L;l++){
		p[l] = randn(0, 1) / sqrt(R / 5);
	}

	return p;

}

double** get_random_matrix(int R, int C)
{
	double ** p;
	p = calloc(R, sizeof(double*));

	if (p == NULL)
		exit(-1);

	for(int r=0;r<R;r++){
		p[r] = calloc(C, sizeof(double));
		if (p[r] == NULL)
			exit(-1);
	}

	for(int r=0;r<R;r++){
		for(int c=0;c<C;c++){
			p[r][c] = ((((double)rand()) / RAND_MAX)) / sqrt(R / 2.0);
		}
	}

	return p;
}

double** get_zero_matrix(int R, int C)
{
	double ** p;
	p = calloc(R, sizeof(double*));

	if (p == NULL)
		exit(-1);

	for(int r=0;r<R;r++){
		p[r] = calloc(C, sizeof(double));
		if (p[r] == NULL)
			exit(-1);
	}

	for(int r=0;r<R;r++) {
		for(int c=0;c<C;c++){
			p[r][c] = 0.0;
		}
	}

	return p;
}

int init_zero_matrix(double*** A, int R, int C)
{
	*A = calloc(R, sizeof(double*));

	if (*A == NULL)
		return -1;

	for(int r=0;r<R;r++){
		(*A)[r] = calloc(C, sizeof(double));
		if ((*A)[r] == NULL)
			return -2;
	}

	for(int r=0;r<R;r++){
		for(int c=0;c<C;c++) {
			(*A)[r][c] = 0.0;
		}
	}

	return 0;
}

int free_matrix(double** A, int R)
{
	for(int r=0;r<R;r++) {
		free(A[r]);
	}
	free(A);
	return 0;
}

int init_zero_vector(double** V, int L)
{
	*V = calloc(L, sizeof(double));
	if (*V == NULL)
		return -1;

	for(int l=0;l<L;l++) {
		(*V)[l] = 0.0;
	}

	return 0;
}

double* get_zero_vector(int L)
{
	double *p;
	p = calloc(L, sizeof(double));
	if (p == NULL)
		exit(0);

	for(int l=0;l<L;l++){
		p[l] = 0.0;
	}

	return p;
}

int free_vector(double** V)
{
	free(*V);
	*V = NULL;
	return 0;
}

void copy_vector(double* A, double* B, int L)
{
	memcpy(A, B, sizeof(double)*L);
}

void matrix_add(double** A, double** B, int R, int C)
{
	for(int r=0;r<R;r++){
		for(int c=0;c<C;c++){
			A[r][c] += B[r][c];
		}
	}
}

void vector_set_to_zero(double* V, int L)
{
	memset(V, 0, sizeof(double)*L);
}


void matrix_set_to_zero(double** A, int R, int C)
{
	for(int r=0;r<R;r++) {
		for(int c=0;c<C;c++) {
			A[r][c] = 0.0;
		}
	}
}

void matrix_substract(double** A, double** B, int R, int C)
{
	for(int r=0;r<R;r++) {
		for (int c = 0; c<C; c++) {
			A[r][c] -= B[r][c];
		}
	}
}

void matrix_scalar_multiply(double** A, double b, int R, int C)
{
	for(int r=0;r<R;r++) {
		for(int c=0;c<C;c++) {
			A[r][c] *= b;
		}
	}
}
void matrix_clip(double** A, double limit, int R, int C)
{
	for(int r=0;r<R;r++) {
		for(int c=0;c<C;c++){
			if (A[r][c] > limit)
				A[r][c] = limit;
			else if (A[r][c] < -limit)
				A[r][C] = -limit;
		}
	}
}

double one_norm(double* V, int L)
{
	double norm = 0.0;
	for(int l=0;l<L;l++) {
		norm += fabs(V[l]);
	}
	return norm;
}

int vectors_fit(double* V, double limit, int L)
{
	int msg = 0;
	double norm;
	for(int l=0;l<L;l++) {
		if (V[l] > limit || V[l] < -limit) {
			msg = 1;
			norm = one_norm(V, L);
			break;
		}
	}

	if (msg)
		vectors_mutliply_scalar(V, limit / norm, L);

	return msg;
}

int vectors_clip(double* V, double limit, int L)
{
	int msg = 0;
	for(int l=0;l<L;l++) {
		if (V[l] > limit) {
			msg = 1;
			V[l] = limit;
		}
		else if (V[l] < -limit) {
			msg = 1;
			V[l] = -limit;
		}
	}

	return msg;
}

void matrix_store(double ** A, int R, int C, FILE * fp)
{
	char * p;

	for(int r=0;r<R;r++) {
		for(int c=0;c<C;c++) {
			p = (char*)&A[r][c];
			for(size_t i=0;i<sizeof(double);i++) {
				fputc(*((char*)p), fp);
				++p;
			}
		}
	}

}

void vector_print_min_max(char *name, double *V, int L)
{
	double min = 100;
	double max = -100;
	for(int l=0;l<L;l++) {
		if (V[l] > max)
			max = V[l];
		if (V[l] < min)
			min = V[l];
	}
	printf("%s min: %.10lf, max: %.10lf\n", name, min, max);
}

void matrix_read(double ** A, int R, int C, FILE * fp)
{
	char * p;
	double value;

	for(int r=0;r<R;r++) {
		for(int c=0;c<C;c++) {
			p = (char*)&value;
			for(size_t i=0;i<sizeof(double);i++){
				*((char *)p) = fgetc(fp);
				++p;
			}
			A[r][c] = value;
		}
	}

}

void vector_store(double* V, int L, FILE * fp)
{
	char * p;

	for(int l=0;l<L;l++) {
		p = (char*)&V[l];
		for(size_t i=0;i<sizeof(double);i++) {
			fputc(*((char*)p), fp);
			++p;
		}
	}
}

void vector_read(double * V, int L, FILE * fp)
{
	char * p;
	double value;

	for(int l=0;l<L;l++){
		p = (char*)&value;
		for(size_t i=0;i<sizeof(double);i++){
			*((char *)p) = fgetc(fp);
			++p;
		}
		V[l] = value;
	}

}

void vector_store_as_matrix_json(double* V, int R, int C, FILE * fp)
{
	if (fp == NULL)
		return; // No file, nothing to do. 

	fprintf(fp, "[");


	for(int r=0;r<R;r++){

		if (r > 0)
			fprintf(fp, ",");

		fprintf(fp, "[");

		for(int c=0;c<C;c++) {

			if (c > 0)
				fprintf(fp, ",");

			fprintf(fp, "%.15f", V[r*C + c]);

		}

		fprintf(fp, "]");

	}

	fprintf(fp, "]");
}


void vector_store_json(double* V, int L, FILE * fp)
{
	if (fp == NULL)
		return; // No file, nothing to do. 

	fprintf(fp, "[");

	for(int l=0;l<L;l++){

		if (l > 0)
			fprintf(fp, ",");

		fprintf(fp, "%.15f", V[l]);
	}

	fprintf(fp, "]");
}

/*
	Gaussian generator
*/
double
randn(double mu, double sigma)
{
	double U1, U2, W, mult;
	static double X1, X2;
	static int call = 0;

	if (call == 1)
	{
		call = !call;
		return (mu + sigma * (double)X2);
	}

	do
	{
		U1 = -1 + ((double)rand() / RAND_MAX) * 2;
		U2 = -1 + ((double)rand() / RAND_MAX) * 2;
		W = pow(U1, 2) + pow(U2, 2);
	} while (W >= 1 || W == 0);

	mult = sqrt((-2 * log(W)) / W);
	X1 = U1 * mult;
	X2 = U2 * mult;

	call = !call;

	return (mu + sigma * (double)X1);
}

double sample_normal() {
	double u = ((double)rand() / (RAND_MAX)) * 2 - 1;
	double v = ((double)rand() / (RAND_MAX)) * 2 - 1;
	double r = u * u + v * v;
	if (r == 0 || r > 1) return sample_normal();
	double c = sqrt(-2 * log(r) / r);
	return u * c;
}
