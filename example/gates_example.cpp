#include <neuralnetwork.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h> // printf için

int main()
{		
	// 2 girişli XOR
	
	DATA_COUNT = 4;
	IN_NEURON_COUNT = 2;
	HID_NEURON_COUNT = 3;
	OUT_NEURON_COUNT = 1;
	ITERATION_COUNT = 600;
	
	
	setupNetwork();
	
	float in1x[] = {0, 0}; float out1x[] = {0};
	float in2x[] = {0, 1}; float out2x[] = {1};
	float in3x[] = {1, 0}; float out3x[] = {1};
	float in4x[] = {1, 1}; float out4x[] = {0};
	
	for (int i = 0; i < ITERATION_COUNT; i++)
	{
		train(in1x, out1x);
		train(in2x, out2x);
		train(in3x, out3x);
		train(in4x, out4x);
		
		calculateDeviations(in1x, out1x, i, 0);
		calculateDeviations(in2x, out2x, i, 1);
		calculateDeviations(in3x, out3x, i, 2);
		calculateDeviations(in4x, out4x, i, 3);
		
		deviations[i] = .0;
		for (int j = 0; j < DATA_COUNT; j++) deviations[i] += dataDeviations[i][j];
		deviations[i] = deviations[i] / DATA_COUNT;
		
		if (i == ITERATION_COUNT-1)
		{		
			printf("\n\nA_2_XOR = [");
			for (int k = 0; k < ITERATION_COUNT; k++) printf("%.4f ", deviations[k]);
			printf("]\n\n");
			
			calculateOuts(in1x); printf("%.4f\n", nOuts[0]);	
			calculateOuts(in2x); printf("%.4f\n", nOuts[0]);	
			calculateOuts(in3x); printf("%.4f\n", nOuts[0]);	
			calculateOuts(in4x); printf("%.4f\n", nOuts[0]);		
		}
	}
	
	// 3 girişli XOR
	
	DATA_COUNT = 8;
	IN_NEURON_COUNT = 3;
	HID_NEURON_COUNT = 4;
	OUT_NEURON_COUNT = 1;
	ITERATION_COUNT = 1300;
	
	
	setupNetwork();
	
	float in1y[] = {0, 0, 0}; float out1y[] = {0};
	float in2y[] = {0, 0, 1}; float out2y[] = {1};
	float in3y[] = {0, 1, 0}; float out3y[] = {1};
	float in4y[] = {0, 1, 1}; float out4y[] = {0};
	float in5y[] = {1, 0, 0}; float out5y[] = {1};
	float in6y[] = {1, 0, 1}; float out6y[] = {0};
	float in7y[] = {1, 1, 0}; float out7y[] = {0};
	float in8y[] = {1, 1, 1}; float out8y[] = {1};			
	
	for (int i = 0; i < ITERATION_COUNT; i++)
	{
		train(in1y, out1y);
		train(in2y, out2y);
		train(in3y, out3y);
		train(in4y, out4y);
		train(in5y, out5y);
		train(in6y, out6y);
		train(in7y, out7y);
		train(in8y, out8y);
		
		calculateDeviations(in1y, out1y, i, 0);
		calculateDeviations(in2y, out2y, i, 1);
		calculateDeviations(in3y, out3y, i, 2);
		calculateDeviations(in4y, out4y, i, 3);
		calculateDeviations(in5y, out5y, i, 4);
		calculateDeviations(in6y, out6y, i, 5);
		calculateDeviations(in7y, out7y, i, 6);
		calculateDeviations(in8y, out8y, i, 7);
		
		deviations[i] = .0;
		for (int j = 0; j < DATA_COUNT; j++) deviations[i] += dataDeviations[i][j];
		deviations[i] = deviations[i] / DATA_COUNT;
		
		if (i == ITERATION_COUNT-1)
		{		
			printf("\n\nA_3_XOR = [");
			for (int k = 0; k < ITERATION_COUNT; k++) printf("%.4f ", deviations[k]);
			printf("]\n\n");
			
			calculateOuts(in1y); printf("%.4f\n", nOuts[0]);	
			calculateOuts(in2y); printf("%.4f\n", nOuts[0]);	
			calculateOuts(in3y); printf("%.4f\n", nOuts[0]);	
			calculateOuts(in4y); printf("%.4f\n", nOuts[0]);	
			calculateOuts(in5y); printf("%.4f\n", nOuts[0]);	
			calculateOuts(in6y); printf("%.4f\n", nOuts[0]);	
			calculateOuts(in7y); printf("%.4f\n", nOuts[0]);	
			calculateOuts(in8y); printf("%.4f\n", nOuts[0]);		
		}
	}
	
	// TAM TOPLAYICI
	
	DATA_COUNT = 8;
	IN_NEURON_COUNT = 3;
	HID_NEURON_COUNT = 6;
	OUT_NEURON_COUNT = 2;
	ITERATION_COUNT = 300;
	
	setupNetwork();
	
	float in1[] = {0, 0, 0}; float out1[] = {0, 0};
	float in2[] = {0, 0, 1}; float out2[] = {1, 0};
	float in3[] = {0, 1, 0}; float out3[] = {1, 0};
	float in4[] = {0, 1, 1}; float out4[] = {0, 1};
	float in5[] = {1, 0, 0}; float out5[] = {1, 0};
	float in6[] = {1, 0, 1}; float out6[] = {0, 1};
	float in7[] = {1, 1, 0}; float out7[] = {0, 1};
	float in8[] = {1, 1, 1}; float out8[] = {1, 1};			
	
	for (int i = 0; i < ITERATION_COUNT; i++)
	{
		train(in1, out1);
		train(in2, out2);
		train(in3, out3);
		train(in4, out4);
		train(in5, out5);
		train(in6, out6);
		train(in7, out7);
		train(in8, out8);
		
		calculateDeviations(in1, out1, i, 0);
		calculateDeviations(in2, out2, i, 1);
		calculateDeviations(in3, out3, i, 2);
		calculateDeviations(in4, out4, i, 3);
		calculateDeviations(in5, out5, i, 4);
		calculateDeviations(in6, out6, i, 5);
		calculateDeviations(in7, out7, i, 6);
		calculateDeviations(in8, out8, i, 7);
		
		deviations[i] = .0;
		for (int j = 0; j < DATA_COUNT; j++) deviations[i] += dataDeviations[i][j];
		deviations[i] = deviations[i] / DATA_COUNT;
		
		if (i == ITERATION_COUNT-1)
		{		
			printf("\n\nA_FULL_ADDER = [");
			for (int k = 0; k < ITERATION_COUNT; k++) printf("%.4f ", deviations[k]);
			printf("]\n\n");
			
			calculateOuts(in1); printf("%.4f	%.4f\n", nOuts[0], nOuts[1] );	
			calculateOuts(in2); printf("%.4f	%.4f\n", nOuts[0], nOuts[1] );	
			calculateOuts(in3); printf("%.4f	%.4f\n", nOuts[0], nOuts[1] );	
			calculateOuts(in4); printf("%.4f	%.4f\n", nOuts[0], nOuts[1] );	
			calculateOuts(in5); printf("%.4f	%.4f\n", nOuts[0], nOuts[1] );	
			calculateOuts(in6); printf("%.4f	%.4f\n", nOuts[0], nOuts[1] );	
			calculateOuts(in7); printf("%.4f	%.4f\n", nOuts[0], nOuts[1] );	
			calculateOuts(in8); printf("%.4f	%.4f\n", nOuts[0], nOuts[1] );		
		}
	}
	
	// 4 Girişli AND
	
	DATA_COUNT = 16;
	IN_NEURON_COUNT = 4;
	HID_NEURON_COUNT = 2;
	OUT_NEURON_COUNT = 1;
	ITERATION_COUNT = 400;
	
	setupNetwork();
	
	float in01[] = {0, 0, 0, 0}; float out01[] = {0};
	float in02[] = {0, 0, 0, 1}; float out02[] = {0};
	float in03[] = {0, 0, 1, 0}; float out03[] = {0};
	float in04[] = {0, 0, 1, 1}; float out04[] = {0};
	float in05[] = {0, 1, 0, 0}; float out05[] = {0};
	float in06[] = {0, 1, 0, 1}; float out06[] = {0};
	float in07[] = {0, 1, 1, 0}; float out07[] = {0};
	float in08[] = {0, 1, 1, 1}; float out08[] = {0};	
	float in11[] = {1, 0, 0, 0}; float out11[] = {0};
	float in12[] = {1, 0, 0, 1}; float out12[] = {0};
	float in13[] = {1, 0, 1, 0}; float out13[] = {0};
	float in14[] = {1, 0, 1, 1}; float out14[] = {0};
	float in15[] = {1, 1, 0, 0}; float out15[] = {0};
	float in16[] = {1, 1, 0, 1}; float out16[] = {0};
	float in17[] = {1, 1, 1, 0}; float out17[] = {0};
	float in18[] = {1, 1, 1, 1}; float out18[] = {1};
	
	for (int i = 0; i < ITERATION_COUNT; i++)
	{
		train(in01, out01);
		train(in02, out02);
		train(in03, out03);
		train(in04, out04);
		train(in05, out05);
		train(in06, out06);
		train(in07, out07);
		train(in08, out08);
		train(in11, out11);
		train(in12, out12);
		train(in13, out13);
		train(in14, out14);
		train(in15, out15);
		train(in16, out16);
		train(in17, out17);
		train(in18, out18);
		
		calculateDeviations(in01, out01, i, 0);
		calculateDeviations(in02, out02, i, 1);
		calculateDeviations(in03, out03, i, 2);
		calculateDeviations(in04, out04, i, 3);
		calculateDeviations(in05, out05, i, 4);
		calculateDeviations(in06, out06, i, 5);
		calculateDeviations(in07, out07, i, 6);
		calculateDeviations(in08, out08, i, 7);
		calculateDeviations(in11, out11, i, 8);
		calculateDeviations(in12, out12, i, 9);
		calculateDeviations(in13, out13, i, 10);
		calculateDeviations(in14, out14, i, 11);
		calculateDeviations(in15, out15, i, 12);
		calculateDeviations(in16, out16, i, 13);
		calculateDeviations(in17, out17, i, 14);
		calculateDeviations(in18, out18, i, 15);
		
		deviations[i] = .0;
		for (int j = 0; j < DATA_COUNT; j++) deviations[i] += dataDeviations[i][j];
		deviations[i] = deviations[i] / DATA_COUNT;
		
		if (i == ITERATION_COUNT-1)
		{		
			printf("\n\nA_4_AND = [");
			for (int k = 0; k < ITERATION_COUNT; k++) printf("%.4f ", deviations[k]);
			printf("]\n\n");
			
			printf("[");
			calculateOuts(in01); printf("%.4f ", nOuts[0]);	
			calculateOuts(in02); printf("%.4f ", nOuts[0]);	
			calculateOuts(in03); printf("%.4f ", nOuts[0]);	
			calculateOuts(in04); printf("%.4f ", nOuts[0]);	
			calculateOuts(in05); printf("%.4f ", nOuts[0]);	
			calculateOuts(in06); printf("%.4f ", nOuts[0]);	
			calculateOuts(in07); printf("%.4f ", nOuts[0]);	
			calculateOuts(in08); printf("%.4f ", nOuts[0]);
			calculateOuts(in11); printf("%.4f ", nOuts[0]);	
			calculateOuts(in12); printf("%.4f ", nOuts[0]);	
			calculateOuts(in13); printf("%.4f ", nOuts[0]);	
			calculateOuts(in14); printf("%.4f ", nOuts[0]);	
			calculateOuts(in15); printf("%.4f ", nOuts[0]);	
			calculateOuts(in16); printf("%.4f ", nOuts[0]);	
			calculateOuts(in17); printf("%.4f ", nOuts[0]);	
			calculateOuts(in18); printf("%.4f ", nOuts[0]);
			printf("]");
		}
	}
	
return 0;	
}
