#include <stdio.h>

#include "lstm.h"
#include "layers.h"
#include "embedding.h"
#include "utilities.h"

#define ITERATIONS 	50000
#define FEATURES 42

// , *layer1 = NULL, *layer2 = NULL;
lstm_model_t *model = NULL;
lstm_model_t **model_layers;

void store_the_net_layers(void)
{
	if (model_layers != NULL) {
		lstm_store_net_layers(model_layers, STD_LOADABLE_NET_NAME);
		lstm_store_net_layers_as_json(model_layers, STD_JSON_NET_NAME);
		printf("\nStored the net\n");
	}
	else {
		printf("\nFailed to store the net!\n");
		exit(-1);
	}

	exit(0);
	return;
}

int main(int argc, char **argv) {
	int p = 0;
	size_t num_datapoints = 0;
	int *X_train, *Y_train;
	
	int layers = 1;
	int neurons = 20;

	lstm_model_parameters_t params;

	params.loss_moving_avg = LOSS_MOVING_AVG;
	params.learning_rate = STD_LEARNING_RATE;
	params.momentum = STD_MOMENTUM;
	params.lambda = STD_LAMBDA;
	params.softmax_temp = SOFTMAX_TEMP;
	params.mini_batch_size = MINI_BATCH_SIZE;
	params.gradient_clip_limit = GRADIENT_CLIP_LIMIT;
	params.learning_rate_decrease = STD_LEARNING_RATE_DECREASE;
	params.learning_rate_decrease_threshold = STD_LEARNING_RATE_THRESHOLD;

	params.beta1 = 0.9;
	params.beta2 = 0.999;

	params.gradient_fit = GRADIENTS_FIT;
	params.gradient_clip = GRADIENTS_CLIP;
	params.model_regularize = MODEL_REGULARIZE;
	params.layers = layers;
	params.optimizer = OPTIMIZE_ADAM;

	srand(time(NULL));

	if (argc < 2) {
		printf("Usage: %s datafile -l layers -n neurons \n", argv[0]);
		return -1;
	}

	// parse params 
	for (int arg = 0; arg < argc; arg++) {
		// number of layers 
		if (!strcmp(argv[arg], "-l")) {
			if (argc >= arg + 1) {
				layers = atoi(argv[arg + 1]);
			} else {
				layers = -1;
			}

			if(layers<1 || layers>10){
				printf("Invalid layer[1~10]\n");
				return -1;
			}
		}

		// number of neurons in a layer 
		if (!strcmp(argv[arg], "-n")) {
			if (argc >= arg + 1) {
				neurons = atoi(argv[arg + 1]);
			} else {
				neurons = -1;
			}

			if (neurons < 1 || neurons > 1024) {
				printf("Invalid neurons[1~1024]\n");
				return -1;
			}
		}
	}

	// call load_and_build_data to allocate and build X_train 
	num_datapoints = load_and_build_data(argv[1], &X_train);
	if (num_datapoints < 0) {
		printf("failed to load and build dataset\n");
		return -1;
	}
	//y_t = x_(t+1)
	Y_train = &X_train[1]; 

	// allocate layers 
	model_layers = (lstm_model_t**) calloc(layers, sizeof(lstm_model_t*));
	if (model_layers == NULL) {
		printf("Error in init!\n");
		exit(-1);
	}

	// initialize per-layer model 
	for (p = 0; p < layers; p++) {
		lstm_init_model(FEATURES, neurons, &model_layers[p], 
			YES_FILL_IT_WITH_A_BUNCH_OF_RANDOM_NUMBERS_PLEASE, &params);
	}

	printf("LSTM Neural net compiled: %s %s, %d Layers, Neurons: %d, "
			"Backprop Through Time: %d, LR: %lf, Mo: %lf, LA: %lf, "
			"LR-decrease: %lf\n", __DATE__, __TIME__, layers, neurons, 
			MINI_BATCH_SIZE, params.learning_rate, params.momentum, 
			params.lambda, params.learning_rate_decrease);

	{
		// train configurated layers with training data 
		lstm_train(
			model_layers[0],
			model_layers,
			num_datapoints,
			X_train,
			Y_train,
			ITERATIONS,
			layers
		);

		// write trained model 
		store_the_net_layers();
	}

	// X_train is allocated by load_and_build_data
	free_loaded_data(X_train);
	free(model_layers);
	
	return 0;
}

