// IRIS_DEMO.C
// Simple Backpropagation Network Demo
// CS470/670 Intro to AI - Marc Pomplun, UMass Boston

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define EXEMPLARS 100		 // total number of exemplars
#define TRAIN_EXEMPLARS 80
#define ERROR_THRESH 0.02
#define RUNS 10

struct EXEMPLAR 
{
	double x1, x2, x3, x4, y;
	int check, inTrainSet;   // arrays used for exemplar order randomization and train/test separation
};

// network variables
double x1, x2, x3, x4, y, h1, h2, o, wh10, wh11, wh12, wh13, wh14, wh20, wh21, wh22, wh23, wh24, wo0, wo1, wo2;
double eta = 1.0;
struct EXEMPLAR exem[EXEMPLARS];


// returns random number between -1 and 1
double randomWeight()
{
	return (double)rand()*2.0 / (double)RAND_MAX - 1.0;
}


// initializes all network weights with random values
void initWeights() {
	wh10 = randomWeight();
	wh11 = randomWeight();
	wh12 = randomWeight();
	wh13 = randomWeight();
	wh14 = randomWeight();
	wh20 = randomWeight();
	wh21 = randomWeight();
	wh22 = randomWeight();
	wh23 = randomWeight();
	wh24 = randomWeight();
	wo0 = randomWeight();
	wo1 = randomWeight();
	wo2 = randomWeight();
}


// read the exemplars from the file and scale them for NN training
void readData() 
{
	double tx1, tx2, tx3, tx4;
	int i = 0;
	char ty[20];
	FILE *f;
	errno_t e;

	e = fopen_s(&f, "iris.txt", "rt");
	for (i = 0; i < EXEMPLARS; i++)
	{
		fscanf_s(f, "%lf %lf %lf %lf %s", &tx1, &tx2, &tx3, &tx4, ty, 20);
		exem[i].x1 = (tx1 - 4.8) / 3.2;
		exem[i].x2 = (tx2 - 1.9) / 2.0;
		exem[i].x3 = (tx3 - 2.9) / 4.1;
		exem[i].x4 = (tx4 - 0.9) / 1.7;
		if (!strcmp(ty, "versicolor"))
			exem[i].y = 0.05;
		else
			exem[i].y = 0.95;
	}
	fclose(f);
}


// computes the network's output for given exemplar but does no weight adjustment
void computeOutput(int ex) 
{
	x1 = exem[ex].x1;
	x2 = exem[ex].x2;
	x3 = exem[ex].x3;
	x4 = exem[ex].x4;
	y = exem[ex].y;

	h1 = 1.0 / (1.0 + exp(-wh10 - x1*wh11 - x2*wh12 - x3*wh13 - x4*wh14));
	h2 = 1.0 / (1.0 + exp(-wh20 - x1*wh21 - x2*wh22 - x3*wh23 - x4*wh24));
	o = 1.0 / (1.0 + exp(-wo0 - h1*wo1 - h2*wo2));
}


// applies the backpropagation learning rule to the current network state
void adjustWeights() {
	double f_prime_h1, f_prime_h2, f_prime_o, delta_h1, delta_h2, delta_o;

	f_prime_h1 = h1*(1.0 - h1);
	f_prime_h2 = h2*(1.0 - h2);
	f_prime_o = o*(1.0 - o);

	delta_o = (y - o)*f_prime_o;
	delta_h1 = f_prime_h1*delta_o*wo1;
	delta_h2 = f_prime_h2*delta_o*wo2;

	wh10 += eta*delta_h1;
	wh11 += eta*delta_h1*x1;
	wh12 += eta*delta_h1*x2;
	wh13 += eta*delta_h1*x3;
	wh14 += eta*delta_h1*x4;

	wh20 += eta*delta_h2;
	wh21 += eta*delta_h2*x1;
	wh22 += eta*delta_h2*x2;
	wh23 += eta*delta_h2*x3;
	wh24 += eta*delta_h2*x4;

	wo0 += eta*delta_o;
	wo1 += eta*delta_o*h1;
	wo2 += eta*delta_o*h2;
}


// sets the exemplars' inTrainSet variable to indicate train vs. test set membership 
void pickTrainSet()
{
	int i, r;

	for (i = 0; i < EXEMPLARS; i++)
		exem[i].inTrainSet = 0;
	for (i = 0; i < TRAIN_EXEMPLARS; i++)
	{
		do
			r = rand() % EXEMPLARS;
		while (exem[r].inTrainSet);
		exem[r].inTrainSet = 1;
	}
}


// train for one epoch and return error
double trainEpoch()
{
	int i, r;
	double errorSum = 0.0, MSE;

	for (i = 0; i < EXEMPLARS; i++)
		exem[i].check = 0;

	for (i = 0; i < TRAIN_EXEMPLARS; i++)
	{
		do
			r = rand() % EXEMPLARS;
		while (exem[r].check || !exem[r].inTrainSet);
		exem[r].check = 1;
		computeOutput(r);
		errorSum += (y - o)*(y - o);
		adjustWeights();
	}
	MSE = errorSum / (double)TRAIN_EXEMPLARS;
	return MSE;
}


// test all exemplars from test set and return number of incorrect classifications
int test()
{
	int i, errSum = 0;
	
	for (i = 0; i < EXEMPLARS; i++)
		if (!exem[i].inTrainSet)
		{
			computeOutput(i);
			if ((o <= 0.5 && y > 0.5) || (o > 0.5 && y <= 0.5))
				errSum++;
		}
	return errSum;
}


void main() {
	int i, epoch;
	double netError;
	int errorSum = 0;

	srand((unsigned)time(NULL));
	readData();

	for (i = 0; i < RUNS; i++)
	{
		printf("\nRUN %d:\n", i + 1);
		pickTrainSet();
		initWeights();
		epoch = 1;
		do
		{
			netError = trainEpoch();
			printf("Epoch %d: %lf\n", epoch++, netError);
		}
		while (netError > ERROR_THRESH);
		errorSum += test();
	}
	printf("Average accuracy: %.2f percent.\n", 100.0 - 100.0*(double)errorSum / (double)(RUNS*(EXEMPLARS - TRAIN_EXEMPLARS)));
}
	