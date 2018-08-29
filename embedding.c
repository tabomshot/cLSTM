#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "embedding.h"

#define BUFFER_LEN 32

// one number per line
int load_and_build_data(const char *filename, int **ptr_arr_data, 
    int op_offset /* optional, feature offset */)
{
    size_t s = 0, i = 0;
    int num_datapoints = 0;
    char c[2] = {0, };
    char buf[BUFFER_LEN] = {0, };
    int *ptr_temp;
    FILE *fp;

    if (*ptr_arr_data) {
        printf("*ptr_arr_data must point NULL!\n");
        return -1;
    }

    fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("Could not open file: %s\n", filename);
		return -1;
	}
	
    // estimate the number of datapoints  
    while ((c[s % 2] = fgetc(fp)) != EOF) {
        if (s == 0) {
            ++s;
            continue;
        }

        if (c[s % 2] == '\n' && c[(s - 1) % 2] != '\n') {
            ++num_datapoints;
        }
        ++s;
	}
	fclose(fp);
	
    // discard last data for training purpose
	num_datapoints--; 
    
    // allocate space to load
    (*ptr_arr_data) = (int *) calloc(num_datapoints + 1, sizeof(int));
    if (*ptr_arr_data == NULL) {
        printf("allocation error\n");
        return -1;
    }
    ptr_temp = *ptr_arr_data;

    // read, parse, put
    fp = fopen(filename, "r");

    // read, parse, put
    s = i = 0;
    memset(buf, 0x00, BUFFER_LEN);
    while ((c[0] = fgetc(fp)) != EOF) {
        if (c[0] == '\n' && s == 0) {
            continue;
        }

        if (c[0] == '\n' && s != 0) {
            buf[s] = 0; // add null
            printf("i: %d\n", i);
            ptr_temp[i++] = atoi(buf) + op_offset;
            memset(buf, 0x00, BUFFER_LEN);
            s = 0;
        } else {
            buf[s++] = c[0];
        }
    }
        
    return num_datapoints;
}
