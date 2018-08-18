CC = gcc

SOURCES = test.c lstm.c embedding.c layers.c utilities.c 
OBJECTS = test.o lstm.o embedding.o layers.o utilities.o 
CFLAGS = -Wall -O3
LDFLAGS = -lm  

all: lstmtest

lstmtest: $(OBJECTS) 
	$(CC) -o $@ $(OBJECTS) $(CFLAGS) $(LDFLAGS) 

clean:
	-rm -f $(OBJECTS) lstmtest

.PHONY: all clean

