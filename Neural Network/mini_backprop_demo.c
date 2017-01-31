// MINI_BACKPROP_DEMO.C
// Simple Backpropagation Network Demo
// CS470/670 Intro to AI - Marc Pomplun, UMass Boston

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define MAX_EXEMPLARS 10		// maximum number of exemplars in the training set

struct EXEMPLAR
{
	double x1, x2, y;
};

double x1, x2, y, h1, h2, o, wh10, wh11, wh12, wh20, wh21, wh22, wo0, wo1, wo2;  // network variables
double eta;														                 // learning step size parameter
int exemplars = 0;												                 // current number of exemplars
int check[MAX_EXEMPLARS];										                 // array used for exemplar order randomization
struct EXEMPLAR trainingSet[MAX_EXEMPLARS];						                 // training set = array of exemplars


// returns random number between -2 and 2
double RandomWeight()
{
	return (double) rand()*4.0/(double) RAND_MAX - 2.0;
}

// initializes all network weights with random values
void InitWeights()
{
	wh10 = RandomWeight();
	wh11 = RandomWeight();
	wh12 = RandomWeight();
	wh20 = RandomWeight();
	wh21 = RandomWeight();
	wh22 = RandomWeight();
	wo0  = RandomWeight();
	wo1  = RandomWeight();
	wo2  = RandomWeight();
}

// lets the user input the exemplars and the value for eta
void EnterData()
{
	int ch;

	do
	{
		printf("\nEnter x1, x2, and y of exemplar #%d: ", exemplars + 1);
		scanf("%lf %lf %lf", &(trainingSet[exemplars].x1), 
			                 &(trainingSet[exemplars].x2), 
							 &(trainingSet[exemplars].y));
		exemplars++;
		printf("\nWould you like to enter another exemplar (y/n) ? ");
		do 
		{
			ch = getch();
		}
		while (ch != 'y' && ch != 'n');
		printf("%c\n", ch);
	}
	while (ch == 'y' && exemplars < 10);
	printf("\nLearning step size parameter eta = ");
	scanf("%lf", &eta);
}


void PrintVarNames()
{
	printf(" x1  x2  wh10  wh11  wh12  h1   wh20  wh21  wh22  h2   wo0   wo1   wo2   o   y\n");
}


void PrintVarValues()
{
	printf("%1.1lf %1.1lf %+1.2lf %+1.2lf %+1.2lf %1.2lf %+1.2lf %+1.2lf %+1.2lf %1.2lf %+1.2lf %+1.2lf %+1.2lf %1.2lf %1.1lf", 
		   x1, x2, wh10, wh11, wh12, h1, wh20, wh21, wh22, h2, wo0, wo1, wo2, o, y);
}

// computes the network's output for given x1, x2, but does no weight adjustment
void ComputeOutput(int exemp)
{
	x1 = trainingSet[exemp].x1;
	x2 = trainingSet[exemp].x2;
	y = trainingSet[exemp].y;
		
	h1 = 1.0/(1.0 + exp(-wh10 - x1*wh11 - x2*wh12));
	h2 = 1.0/(1.0 + exp(-wh20 - x1*wh21 - x2*wh22));
    o = 1.0/(1.0 + exp(-wo0 - h1*wo1 - h2*wo2));
}

// applies the backpropagation learning rule to the current network state
void AdjustWeights()
{
	double f_prime_h1, f_prime_h2, f_prime_o, delta_h1, delta_h2, delta_o;

	f_prime_h1 = h1*(1.0 - h1);
	f_prime_h2 = h2*(1.0 - h2);
	f_prime_o  =  o*(1.0 -  o);

	delta_o = (y - o)*f_prime_o;
	delta_h1 = f_prime_h1*delta_o*wo1;
	delta_h2 = f_prime_h2*delta_o*wo2;

	wh10 += eta*delta_h1;
	wh11 += eta*delta_h1*x1;
	wh12 += eta*delta_h1*x2;
	wh20 += eta*delta_h2;
	wh21 += eta*delta_h2*x1;
	wh22 += eta*delta_h2*x2;

	wo0 += eta*delta_o;
	wo1 += eta*delta_o*h1;
	wo2 += eta*delta_o*h2;
}


void main()
{
	int i, exemp, epoch = 1;

	srand((unsigned) time(NULL));
	InitWeights();
	EnterData();

	printf("\n\nNetwork output before training:\n\n");
	PrintVarNames();
	for (i = 0; i < exemplars; i++)
	{
		ComputeOutput(i);
		PrintVarValues();
	}

	printf("\nPress any key to train next epoch; press 'x' to exit.\n\n");
	while (1)
	{
		if (getch() == 'x')
			exit(0);

		printf("Epoch %d:\n", epoch++);
		PrintVarNames();
		for (i = 0; i < exemplars; i++)
			check[i] = 0;

		for (i = 0; i < exemplars; i++)
		{
			do 
			{
				exemp = rand()%exemplars;
			}
			while (check[exemp]);
			check[exemp] = 1;
			
			ComputeOutput(exemp);
			PrintVarValues();
			AdjustWeights();
		}
		printf("\n");
	}
}
