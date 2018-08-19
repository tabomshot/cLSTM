#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
//#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "utilities.h"
#include "layers.h"
#include "assert.h"

#define STD_LEARNING_RATE										0.001//0.001
#define STD_MOMENTUM											0.01
#define STD_LAMBDA												0.05
#define SOFTMAX_TEMP											1.0
#define GRADIENT_CLIP_LIMIT										5.0
#define MINI_BATCH_SIZE											100 //~60sec
#define LOSS_MOVING_AVG											0.01//0.01


#define STATEFUL												1

// #define INTERLAYER_SIGMOID_ACTIVATION								

#define GRADIENTS_CLIP											1
#define GRADIENTS_FIT											0

//#define DECREASE_LR 

#define MODEL_REGULARIZE										0
	
#define OPTIMIZE_ADAM											1
#define OPTIMIZE_GRADIENT_DESCENT								0

// #define DEBUG_PRINT

#define STD_LEARNING_RATE_DECREASE								100000
#define STD_LEARNING_RATE_THRESHOLD								10000
#define STD_NUMBER_OF_NO_RECORD_ITERATIONS_UNTIL_LR_DECREASE	1000000							

// #define STORE_DURING_TRANING
#define PRINT_EVERY_X_ITERATIONS								100
#define STORE_EVERY_X_ITERATIONS								8000
#define STORE_PROGRESS_EVERY_X_ITERATIONS						1000

#define NUMBER_OF_CHARS_TO_DISPLAY_DURING_TRAINING				10 //upto 1sec

#define YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE				1
#define YES_FILL_IT_WITH_A_BUNCH_OF_RANDOM_NUMBERS_PLEASE		0

#define STD_LOADABLE_NET_NAME									"lstm_net.net"
#define STD_JSON_NET_NAME										"lstm_net.json"
#define PROGRESS_FILE_NAME										"progress.csv"

#define JSON_KEY_NAME_SET										"Feature mapping"
#define ABORT_WHEN_IN_PURGATORY									1

#define USE_CROSS_ENTROPY										0

typedef struct lstm_model_parameters_t {
	// For progress monitoring
	double loss_moving_avg;
	// For gradient descent
	double learning_rate;
	double momentum;
	double lambda;
	double softmax_temp;

	double beta1;
	double beta2;

	int layers;

	int gradient_clip;
	int gradient_fit;

	int optimizer;

	int model_regularize;

	int learning_rate_decrease_threshold;
	double learning_rate_decrease;
	// General parameters
	int mini_batch_size;
	double gradient_clip_limit;
} lstm_model_parameters_t;

typedef struct lstm_model_t
{
		int F; // Number of features
		int N; // Number of neurons
		int S; // The sum of the above..

		// Parameters
		lstm_model_parameters_t * params;

		// The model
		double* Wf;
		double* Wi;
		double* Wc;
		double* Wo;
		double* Wy;
		double* bf;
		double* bi;
		double* bc;
		double* bo;
		double* by;

		// cache
		double* dldh;
		double* dldho;
		double* dldhf;
		double* dldhi;
		double* dldhc;
		double* dldc;

		double* dldXi;
		double* dldXo;
		double* dldXf;
		double* dldXc;

		// Gradient descent momentum
		double* Wfm;
		double* Wim;
		double* Wcm;
		double* Wom;
		double* Wym;
		double* bfm;
		double* bim;
		double* bcm;
		double* bom;
		double* bym;


} lstm_model_t;

typedef struct lstm_values_cache_t {
	double* probs;
	double* probs_before_sigma;
	double* c;
	double* h;
	double* c_old;
	double* h_old;
	double* X;
	double* hf;
	double* hi;
	double* ho;
	double* hc;
	double* tanh_c_cache;
} lstm_values_cache_t;

typedef struct lstm_values_state_t {
	double* c;
	double* h;
} lstm_values_state_t;

typedef struct lstm_values_next_cache_t {
	double* dldh_next;
	double* dldc_next;
	double* dldY_pass;
} lstm_values_next_cache_t;

//					 F,   N,  &lstm model,    zeros, parameters
int lstm_init_model(int, int, lstm_model_t**, int, lstm_model_parameters_t *);
void lstm_zero_the_model(lstm_model_t*);
void lstm_zero_d_next(lstm_values_next_cache_t *, int, int);
void lstm_cache_container_set_start(lstm_values_cache_t *, int);

//					 lstm model to be freed
void lstm_free_model(lstm_model_t*);
//							model, input,  state and cache values, &state and cache values
void lstm_forward_propagate(lstm_model_t*, double*, lstm_values_cache_t*, lstm_values_cache_t*, int);
//							model, y_probabilities, y_correct, the next deltas, state and cache values, &gradients, &the next deltas
void lstm_backward_propagate(lstm_model_t*, double*, int, lstm_values_next_cache_t*, lstm_values_cache_t*, lstm_model_t*, lstm_values_next_cache_t*);

void lstm_values_state_init(lstm_values_state_t** d_next_to_set, int N);
void lstm_values_next_state_free(lstm_values_state_t* d_next);

lstm_values_cache_t*  lstm_cache_container_init(int N, int F);
void lstm_cache_container_free(lstm_values_cache_t*);
void lstm_values_next_cache_init(lstm_values_next_cache_t**, int, int);
void lstm_values_next_cache_free(lstm_values_next_cache_t*);
void sum_gradients(lstm_model_t*, lstm_model_t*);

// For storing and loading of net data
//					model (already init), name
void lstm_read_net_layers(lstm_model_t**, const char*);
void lstm_store_net_layers(lstm_model_t**, const char *);
void lstm_store_net_layers_as_json(lstm_model_t**, int, const char *); 
void lstm_read_net_layers(lstm_model_t**, const char *);
void lstm_store_progress(unsigned int, double);

// The main entry point
//						model, number of training points, X_train, Y_train, number of iterations
void lstm_train(lstm_model_t*, lstm_model_t**,  unsigned int, int*, int*, unsigned long, int);
// Used to output a given number of characters from the net based on an input char
void lstm_output_string_layers(lstm_model_t **,  int, int, int);

#endif
