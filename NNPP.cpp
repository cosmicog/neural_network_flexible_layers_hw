#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h> // printf için

int DATA_COUNT = 8;
int IN_NEURON_COUNT = 3;
int HID_NEURON_COUNT = 6;
int OUT_NEURON_COUNT = 2;
int ITERATION_COUNT = 300;
float LAMBDA = .5f; // Öğrenme katsayısı
float ALPHA = .8f; // Momentum

float ** dataDeviations; // Datalara göre sapmalar      -- dataDeviations[ ITERATION_COUNT ][ DATA_COUNT ]
float * deviations; // Tüm datalara göre roundların sapmalları

float * nIns; // Giriş Nöronlar
float * nHids; // Gizli katman nöronları
float * nOuts; // Çıkış katmanı nöronları
float * nHids_Err; // Gizli katman hatalar
float * nOuts_Err;

float * hidNets; // Gizli katman Net değerleri
float * outNets; // Çıkış katmanı Net değerleri

float ** wHids; // Dentritler, yani ağırlıklar        -- wHids[ HID_NEURON_COUNT ][ IN_NEURON_COUNT ]
float ** dHids; // Delta, yani değişim katsayıları    -- dHids[ HID_NEURON_COUNT ][ IN_NEURON_COUNT ]
float ** wOuts; //                                    -- wOuts[ OUT_NEURON_COUNT ][ HID_NEURON_COUNT ]
float ** dOuts; //                                    -- dOuts[ OUT_NEURON_COUNT ][ HID_NEURON_COUNT ]

float * wHidB; // Sağlama nöronunun ağırlıkları ve 
float * dHidB; // değişim katsayıları
float * wOutB;
float * dOutB;


float randomFloat(float start = .001, float end = .998)
{
	return ( start + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(end))) ); 
	// 0.001 ve 0.999 arasında rastgele sayılar.
}

void setupNetwork()
{
	dataDeviations = new float * [ITERATION_COUNT]; // Datalara göre sapmalar      -- dataDeviations[ ITERATION_COUNT ][ DATA_COUNT ]
	deviations = new float [ITERATION_COUNT]; // Tüm datalara göre roundların sapmalları
	
	nIns = new float [ IN_NEURON_COUNT ]; // Giriş Nöronlar
	nHids = new float [ HID_NEURON_COUNT ]; // Gizli katman nöronları
	nOuts = new float [ OUT_NEURON_COUNT ]; // Çıkış katmanı nöronları
	nHids_Err = new float [ HID_NEURON_COUNT ]; // Gizli katman hatalar
	nOuts_Err = new float [ OUT_NEURON_COUNT ];
	
	hidNets = new float [ HID_NEURON_COUNT ]; // Gizli katman Net değerleri
	outNets = new float [ OUT_NEURON_COUNT ]; // Çıkış katmanı Net değerleri
	
	wHids = new float * [ HID_NEURON_COUNT ]; // Dentritler, yani ağırlıklar        -- wHids[ HID_NEURON_COUNT ][ IN_NEURON_COUNT ]
	dHids = new float * [ HID_NEURON_COUNT ]; // Delta, yani değişim katsayıları    -- dHids[ HID_NEURON_COUNT ][ IN_NEURON_COUNT ]
	wOuts = new float * [ OUT_NEURON_COUNT ]; //                                    -- wOuts[ OUT_NEURON_COUNT ][ HID_NEURON_COUNT ]
	dOuts = new float * [ OUT_NEURON_COUNT ]; //                                    -- dOuts[ OUT_NEURON_COUNT ][ HID_NEURON_COUNT ]
	
	wHidB = new float [HID_NEURON_COUNT]; // Sağlama nöronunun ağırlıkları ve 
	dHidB = new float [HID_NEURON_COUNT]; // değişim katsayıları
	wOutB = new float [OUT_NEURON_COUNT];
	dOutB = new float [OUT_NEURON_COUNT];
	
	// Dinamik dizileri tanımlıyoruz
	for (int i = 0; i < HID_NEURON_COUNT; i++)
	{
		wHids[i] = new float[ IN_NEURON_COUNT ];
		dHids[i] = new float[ IN_NEURON_COUNT ];
	}
	for (int i = 0; i < OUT_NEURON_COUNT; i++)
	{
		wOuts[i] = new float[ HID_NEURON_COUNT ];
		dOuts[i] = new float[ HID_NEURON_COUNT ];
	}
	for (int i = 0; i < ITERATION_COUNT; i++) dataDeviations[i] = new float [DATA_COUNT];
	
	// Dendritlere rastgele değerler veriyoruz ve değişimleri sıfırlıyoruz
	for (int hidc = 0; hidc < HID_NEURON_COUNT; hidc++)
	{
		for (int inc = 0; inc < IN_NEURON_COUNT; inc++)
		{
			wHids[hidc][inc] = randomFloat();
			dHids[hidc][inc] = .0;
		}
		wHidB[hidc] = randomFloat();
		dHidB[hidc] = .0;
	}
	
	for (int outc = 0; outc < OUT_NEURON_COUNT; outc++)
	{
		for (int hidc = 0; hidc < HID_NEURON_COUNT; hidc++)
		{
			wOuts[outc][hidc] = randomFloat();
			dOuts[outc][hidc] = .0;
		}
		wOutB[outc] = randomFloat();
		dOutB[outc] = .0;
	}
}

float activation(float x) // Sigmoid 
{
	return (1.0 / (1.0 + exp(-x)));
	// return (x / (1.0 + abs(x))); //Fast Sigmoid... Sonra bak buna...
}

void calculateOuts(float * inValues) // İleri besleme
{
	for (int i = 0; i < IN_NEURON_COUNT; i++) nIns[i] = inValues[i];
	
	for (int hidc = 0; hidc < HID_NEURON_COUNT; hidc++)
	{
		hidNets[hidc] = .0;
		for (int inc = 0; inc < IN_NEURON_COUNT; inc++)
		{
			hidNets[hidc] += nIns[inc] * wHids[hidc][inc];
		}
		hidNets[hidc] += 1 * wHidB[hidc];
		nHids[hidc] = activation( hidNets[hidc] );
	}
	
	for (int outc = 0; outc < OUT_NEURON_COUNT; outc++)
	{
		outNets[outc] = .0;
		for (int hidc = 0; hidc < HID_NEURON_COUNT; hidc++)
		{
			outNets[outc] += nHids[hidc] * wOuts[outc][hidc];
		}
		outNets[outc] += 1 * wOutB[outc];
		nOuts[outc] = activation( outNets[outc] );
	}
}

void train( float * iVals, float * oVals)
{
	calculateOuts( iVals );
	for (int outc = 0; outc < OUT_NEURON_COUNT; outc++)
	{
		nOuts_Err[outc] = ( oVals[outc] - nOuts[outc] ) * nOuts[outc] * (1 - nOuts[outc]);
		for (int hidc = 0; hidc < HID_NEURON_COUNT; hidc++)
		{
			nHids_Err[hidc] = wOuts[outc][hidc] * nOuts_Err[outc] * nHids[hidc] * (1 - nHids[hidc]);
		}
		for (int hidc = 0; hidc < HID_NEURON_COUNT; hidc++)
		{
			for (int inc = 0; inc < IN_NEURON_COUNT; inc++)
			{
				dHids[hidc][inc] = LAMBDA * nHids_Err[hidc] * nIns[inc] + ALPHA * dHids[hidc][inc];
				wHids[hidc][inc] += dHids[hidc][inc];
			}
			dHidB[hidc] = LAMBDA * nHids_Err[hidc] * 1 + ALPHA * dHidB[hidc];
			wHidB[hidc] = wHidB[hidc] + dHidB[hidc];
		}

		for (int outc = 0; outc < OUT_NEURON_COUNT; outc++)
		{
			for (int hidc = 0; hidc < HID_NEURON_COUNT; hidc++)
			{
				dOuts[outc][hidc] = LAMBDA * nOuts_Err[outc] * nHids[hidc] + ALPHA * dOuts[outc][hidc];
				wOuts[outc][hidc] += dOuts[outc][hidc];
			}
			dOutB[outc] = LAMBDA * nOuts_Err[outc] * 1 + ALPHA * dOutB[outc];
			wOutB[outc] = wOutB[outc] + dOutB[outc];
		}	
	}
}

float errorFactor(float target, float source)
{
	return (float)sqrt(pow(target - source, 2));
}

void calculateDeviations( float * iVals, float * oVals, int iteration_seq, int data_seq)
{
		calculateOuts(iVals);
		dataDeviations[iteration_seq][data_seq] = .0;
		for (int j = 0; j < OUT_NEURON_COUNT; j++)
		{
			dataDeviations[iteration_seq][data_seq] += errorFactor(oVals[j], nOuts[j]);
		}
		dataDeviations[iteration_seq][data_seq] = dataDeviations[iteration_seq][data_seq]/OUT_NEURON_COUNT;
}

main()
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
	
	
}
