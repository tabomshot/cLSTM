#include <stdio.h>

#include "lstm.h"
#include "layers.h"
#include "embedding.h"
#include "utilities.h"

#define MAXBUFSIZE 64

static lstm_model_t **model_layers = NULL;

static int read_jsonelem_int(const char *filename, const char *tag, int *res)
{
	char c;
	int i, s;
	char buf[MAXBUFSIZE];
	FILE *fp; 

	fp = fopen(filename, "r");
	if ( fp == NULL ) {
		printf("Failed to open file: %s\n", filename);
		return -1;
	}

	i = s = 0;
	memset(buf, 0x00, MAXBUFSIZE);
	while ((c = fgetc(fp)) != EOF) {
		if (c == '\"' && s == 0) {
			s = 1;
			continue;
		}

		if (c != '\"' && s == 1) {
			buf[i++] = c;
			continue;
		}

		if (c == '\"' && s == 1) {
			if (!strcmp(buf, tag)) {
				s = 2;
			} else {
				s = 0;
			}
			i = 0;
			memset(buf, 0x00, MAXBUFSIZE);
			continue;
		}

		if (s == 2 && (c == ',' || c == '\n')) {
			break;
		}

		if (s == 2 && c != ':' && c != ' ') {
			buf[i++] = c;
		}
	}

	if (s != 2) {
		printf("Tag %s not found \n", tag);
		return -1;
	}

	printf("Tag %s found: %s\n", tag, buf);
	*res = atoi(buf);

	return 0;
}

static int read_lstm_model(const char *f_dataset_app, const char *f_dataset_locc)
{
	int i;
	int nr_layers, nr_neurons, nr_apps, nr_loccs;
	
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
		/* .layers = nr_layers, ::: init later */ 
		.optimizer = OPTIMIZE_ADAM,
	};	

	// load net metadata from file 
	read_jsonelem_int(STD_JSON_NET_NAME, "LSTM neurons", &nr_neurons);
	read_jsonelem_int(STD_JSON_NET_NAME, "Apps", &nr_apps);
	read_jsonelem_int(STD_JSON_NET_NAME, "Loccs", &nr_loccs);
	read_jsonelem_int(STD_JSON_NET_NAME, "LSTM layers", &nr_layers);
	params.layers = nr_layers;

	// allocate layers 
	model_layers = (lstm_model_t **) calloc(nr_layers, sizeof(lstm_model_t*));
	if (model_layers == NULL) {
		printf("Error: failed to allocate layers\n");
		return -1;
	}

	// initialize per-layer model 
	for (i = 0; i < nr_layers; i++) {
		lstm_init_model(nr_apps + nr_loccs, nr_neurons, &model_layers[i], 
			YES_FILL_IT_WITH_A_BUNCH_OF_RANDOM_NUMBERS_PLEASE, &params);
	}

	// load from file 
	lstm_read_net_layers(model_layers, STD_LOADABLE_NET_NAME);

	return 0;
}

static int run_lstm_prediction(int input_app, int input_locc)
{
	int i, p, nr_layers;
	double *first_layer_input, *last_layer_output;
	lstm_values_cache_t ***cache_layers;
	lstm_values_next_cache_t **d_next_layers;

	if (model_layers == NULL) {
		printf("model layers do not exist\n");
		return -1;
	}
	
	//lstm_forward_propagate(lstm_model_t* model, double * input, 
	//	lstm_values_cache_t* cache_in, lstm_values_cache_t* cache_out, int softmax);

	nr_layers = model_layers[0]->params->layers;

	// initialize 
	init_zero_vector(&first_layer_input, model_layers[0]->F);
	cache_layers = calloc(nr_layers, sizeof(lstm_values_cache_t**));
	if (!cache_layers) {
		printf("Failed to allocate memory for the caches\n");
		return -1;
	}

	for (i = 0; i < nr_layers; i++) {
		cache_layers[i] = calloc(1 /* treat batch as 1*/, 
								sizeof(lstm_values_cache_t*));
		if (!cache_layers[i]) {
			printf("Failed to allocate memory for the caches\n");
			return -1;
		}

		cache_layers[i][0] = lstm_cache_container_init(model_layers[i]->N, model_layers[i]->F);
		if (!cache_layers[i][0]) {
			printf("Failed to allocate memory for the caches\n");
			return -1;
		}
	}

	// make input activations 
	for (i = 0; i < model_layers[0]->F; i++) {
		first_layer_input[i] = 0.0;
	}
	first_layer_input[input_app] = 1.0;
	if (input_locc > 0) {
		first_layer_input[input_locc] = 1.0;
	}
	for (i = 0; i < model_layers[0]->F; i++) {
		printf("F%d: %f,  ", i, first_layer_input[i]);
	}
	printf("\n\n");

	// predict
	p = nr_layers - 1;
	lstm_forward_propagate(model_layers[p], first_layer_input, cache_layers[p][0], cache_layers[p][0], 0);
	last_layer_output = cache_layers[p][0]->probs;
	for (i = 0; i < model_layers[0]->F; i++) {
		printf("F%d: %f,  ", i, last_layer_output[i]);
	}
	printf("\n\n");

	if (p > 0) {
		--p;
		while (p >= 0) {
			// model, input, cache_in, cache_out, softmax? 
			lstm_forward_propagate(model_layers[p], cache_layers[p+1][0]->probs, cache_layers[p][0], cache_layers[p][0], p == 0);	
			for (i = 0; i < model_layers[0]->F; i++) {
				printf("F%d: %f,  ", i, last_layer_output[i]);
			}
			printf("\n\n");
			--p;
		}
		p = 0;
	}

	last_layer_output = cache_layers[p][0]->probs;
	for (i = 0; i < model_layers[0]->F; i++) {
		printf("F%d: %f,  ", i, last_layer_output[i]);
	}
	printf("\n\n");

	return 0;
}

int main(int argc, char **argv) 
{
	read_lstm_model("dataset.set", NULL);
    run_lstm_prediction(1, 0);

	return 0;
}

