CC = gcc

OBJECTS 		= lstm.o embedding.o layers.o utilities.o 
OBJECTS_TRAIN 	= train.o 
OBJECTS_PREDICT = predict.o 
CFLAGS = -Wall -O3 -DNDEBUG 
#CFLAGS = -Wall -g -O0 -DDEBUG
LDFLAGS = -lm  

all: lstmtrain lstmpredict

lstmtrain: $(OBJECTS_TRAIN) $(OBJECTS)
	$(CC) $(OBJECTS_TRAIN) $(OBJECTS) $(CFLAGS) -o $@ $(LDFLAGS) 

lstmpredict: $(OBJECTS_PREDICT) $(OBJECTS)
	$(CC) $(OBJECTS_PREDICT) $(OBJECTS) $(CFLAGS) -o $@ $(LDFLAGS) 

clean:
	-rm -f $(OBJECTS_PREDICT) $(OBJECTS_TRAIN) $(OBJECTS) lstmtrain lstmpredict 

.PHONY: all clean

