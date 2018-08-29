#include <stdio.h>

#include "lstm.h"
#include "layers.h"
#include "embedding.h"
#include "utilities.h"

#define ITERATIONS (10000)

static int store_net_layers(int nr_layers, int nr_apps, int nr_loccs, 
							lstm_model_t **model_layers)
{
	if (model_layers == NULL) {
		printf("Error: model_layers is null!\n");
		return -1;
	}

	lstm_store_net_layers(model_layers, STD_LOADABLE_NET_NAME);
	lstm_store_net_layers_as_json(model_layers, nr_layers, 
		nr_apps, nr_loccs, STD_JSON_NET_NAME);
	printf("model stored\n");

	return 0;
} 

static int run_lstm_train(int nr_layers, int nr_neurons, int nr_apps, 
						int nr_loccs, /* location clusters */ 
						const char *f_dataset_app, const char *f_dataset_locc)
{
	int i = 0;
	int ret1, ret2;
	int nr_features, nr_datapoints; 

	int *X_train_app = NULL;
	int *X_train_locc = NULL;
	int *Y_train_app = NULL;

	lstm_model_t **model_layers = NULL;
	lstm_model_parameters_t params = {
		.loss_moving_avg = LOSS_MOVING_AVG,
		.learning_rate = STD_LEARNING_RATE,
		.momentum = STD_MOMENTUM,
		.lambda = STD_LAMBDA,
		.softmax_temp = SOFTMAX_TEMP,
		.mini_batch_size = MINI_BATCH_SIZE,
		.gradient_clip_limit = GRADIENT_CLIP_LIMIT,
		.learning_rate_decrease = STD_LEARNING_RATE_DECREASE,
		.learning_rate_decrease_threshold = STD_LEARNING_RATE_THRESHOLD,
		.beta1 = 0.9,
		.beta2 = 0.999,
		.gradient_fit = GRADIENTS_FIT,
		.gradient_clip = GRADIENTS_CLIP,
		.model_regularize = MODEL_REGULARIZE,
		.layers = nr_layers,
		.optimizer = OPTIMIZE_ADAM,
	};	

	srand(time(NULL));

	// call load_and_build_data to allocate and build X_train 
	ret1 = load_and_build_data(f_dataset_app, &X_train_app, 0);
	if (ret1 < 0) {
		printf("Error: failed to load and build dataset\n");
		return -1;
	}
	nr_datapoints = ret1; 
	nr_features = nr_apps;

	if (nr_loccs > 0) {
		ret2 = load_and_build_data(f_dataset_locc, &X_train_locc, nr_apps);
		if (ret2 < 0 || ret2 != ret1) {
			printf("Error: location datapoints is not equal to app\n");
			return -1;
		}
		nr_features += nr_loccs;
	}

	//y_t = x_(t+1)
	Y_train_app = &X_train_app[1];

	// allocate layers 
	model_layers = (lstm_model_t**) calloc(nr_layers, sizeof(lstm_model_t*));
	if (model_layers == NULL) {
		printf("Error: failed to allocate layers\n");
		return -1;
	}

	// initialize per-layer model 
	for (i = 0; i < nr_layers; i++) {
		lstm_init_model(nr_features, nr_neurons, &model_layers[i], 
			YES_FILL_IT_WITH_A_BUNCH_OF_RANDOM_NUMBERS_PLEASE, &params);
	}

	printf("LSTM Neural net compiled: %s %s, %d Layers, Neurons: %d, "
		"Backprop Through Time: %d, LR: %lf, Mo: %lf, LA: %lf, "
		"LR-decrease: %lf\n", __DATE__, __TIME__, nr_layers, nr_neurons, 
		MINI_BATCH_SIZE, params.learning_rate, params.momentum, 
		params.lambda, params.learning_rate_decrease);

	// train configurated layers with training data 
	lstm_train(model_layers[0], model_layers, nr_datapoints, 
			X_train_app, X_train_locc, Y_train_app, ITERATIONS, nr_layers);

	// store trained model 
	store_net_layers(nr_layers, nr_apps, nr_loccs, model_layers);

	// free memory areas
	if (X_train_app) 
		free(X_train_app);
	if (X_train_locc) 
		free(X_train_locc);
	if (model_layers)
		free(model_layers);

	return 0;
}

int main(int argc, char **argv) 
{
	int nr_layers = 1;
	int nr_neurons = 20;
	char *f_dataset_app = NULL, *f_dataset_locc = NULL;
	
	if (argc < 7) {
		printf("Usage: %s -d dataset_app -p dataset_locc -l layers -n neurons \n", argv[0]);
		return -1;
	}

	// parse params 
	for (int arg = 0; arg < argc; arg++) {
		// application usage data
		if (!strcmp(argv[arg], "-d")) {
			if (argc > arg + 1) {
				f_dataset_app = argv[arg + 1];
			}
		}

		// location history data
		if (!strcmp(argv[arg], "-p")) {
			if (argc > arg + 1) {
				f_dataset_locc = argv[arg + 1];
			}
		}

		// number of layers 
		if (!strcmp(argv[arg], "-l")) {
			if (argc >= arg + 1) {
				nr_layers = atoi(argv[arg + 1]);
			} else {
				nr_layers = -1;
			}

			if(nr_layers < 1 || nr_layers > 10){
				printf("Invalid layer[1~10]\n");
				return -1;
			}
		}

		// number of neurons in a layer 
		if (!strcmp(argv[arg], "-n")) {
			if (argc >= arg + 1) {
				nr_neurons = atoi(argv[arg + 1]);
			} else {
				nr_neurons = -1;
			}

			if (nr_neurons < 1 || nr_neurons > 1024) {
				printf("Invalid neurons[1~1024]\n");
				return -1;
			}
		}
	}

	if (f_dataset_app == NULL) {
		printf("Usage: %s -d appdata -p locdata -l layers -n neurons \n", argv[0]);
		return -1;
	}

	run_lstm_train(nr_layers, nr_neurons, 42, 0,  f_dataset_app, f_dataset_locc);

	return 0;
}

