///////////////////////////////////////////////////
// Low Level Parallel Programming 2017.
//
// 
//
// The main starting point for the crowd simulation.
//



#undef max
#include "ped_model.h"
#include "MainWindow.h"
#include "ParseScenario.h"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QApplication>
#include <QTimer>
#include <thread>

#include "PedSimulation.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstring>

#pragma comment(lib, "libpedsim.lib")

#include <stdlib.h>

int main(int argc, char*argv[]) {
	bool timing_mode = 0;
	int num_threads = 1;
	int i = 1;
	double split_factor = 1.5;
	double merge_factor = 0.3;
	// Change this variable when testing different versions of your code. 
	// May need modification or extension in later assignments depending on your implementations
	Ped::IMPLEMENTATION implementation_to_test = Ped::SEQ;

	QString scenefile = "scenario.xml";

	// Argument handling
	while (i < argc)
	{
		if (argv[i][0] == '-' && argv[i][1] == '-')
		{
			if (strcmp(&argv[i][2], "timing-mode") == 0)
			{
				cout << "Timing mode on\n";
				timing_mode = true;
			}
			else if (strcmp(&argv[i][2], "help") == 0)
			{
				cout << "Usage: " << argv[0] << " [--help] [--timing-mode] [--threads X (number of threads)] [--split-factor X] [--merge-factor X] [-impl implementation (SEQ, OMP, PTHREAD, VECTOR, CUDA, SEQ1, SEQ2, MOVE_SEQ, MOVE_CONSTANT, MOVE_ADAPTIVE, SEQ_HEATMAP, CUDA_HEATMAP)] [scenario]" << endl;
				return 0;
			}
			else if (strcmp(&argv[i][2], "threads") == 0) {
				i++;
				num_threads = atoi(argv[i]);
				std::cout << "Number of threads set: " << num_threads << std::endl;
			}
			else if (strcmp(&argv[i][2], "split-factor") == 0) {
				i++;
				split_factor = atof(argv[i]);
				std::cout << "Split factor: " << split_factor << std::endl;
			}
			else if (strcmp(&argv[i][2], "merge-factor") == 0) {
				i++;
				merge_factor = atof(argv[i]);
				std::cout << "Merge factor: " << merge_factor << std::endl;
			}			
			else if (strcmp(&argv[i][2], "impl") == 0) {
				i++;
				if (strcmp(argv[i], "SEQ") == 0)
				{
					implementation_to_test = Ped::SEQ;
					std::cout << "Implementation: SEQ" << std::endl;
				}
				else if (strcmp(argv[i], "OMP") == 0)
				{
					implementation_to_test = Ped::OMP;
					std::cout << "Implementation: OpenMp" << std::endl;

				}
				else if (strcmp(argv[i], "PTHREAD") == 0)
				{
					implementation_to_test = Ped::PTHREAD;
					std::cout << "Implementation: PTHREAD" << std::endl;
				}
				else if (strcmp(argv[i], "VECTOR") == 0)
				{
					implementation_to_test = Ped::VECTOR;
					std::cout << "Implementation: VECTOR" << std::endl;
				}
				else if (strcmp(argv[i], "CUDA") == 0)
				{
					implementation_to_test = Ped::CUDA;
					std::cout << "Implementation: CUDA" << std::endl;
				}
				else if (strcmp(argv[i], "SEQ1") == 0)
				{
					implementation_to_test = Ped::SEQ1;
					std::cout << "Implementation: SEQ1" << std::endl;
				}
				else if (strcmp(argv[i], "SEQ2") == 0)
				{
					implementation_to_test = Ped::SEQ;
					std::cout << "Implementation: SEQ2" << std::endl;
				}
				else if (strcmp(argv[i], "MOVE_SEQ") == 0)
				{
					implementation_to_test = Ped::MOVE_SEQ;
					std::cout << "Implementation: MOVE_SEQ" << std::endl;
				}
				else if (strcmp(argv[i], "MOVE_CONSTANT") == 0)
				{
					implementation_to_test = Ped::MOVE_CONSTANT;
					std::cout << "Implementation: MOVE_CONSTANT" << std::endl;
				}
				else if (strcmp(argv[i], "MOVE_ADAPTIVE") == 0)
				{
					implementation_to_test = Ped::MOVE_ADAPTIVE;
					std::cout << "Implementation: MOVE_ADAPTIVE" << std::endl;
				}
				else if (strcmp(argv[i], "SEQ_HEATMAP") == 0)
				{
					implementation_to_test = Ped::SEQ_HEATMAP;
					std::cout << "Implementation: SEQ_HEATMAP" << std::endl;
				}
				else if (strcmp(argv[i], "CUDA_HEATMAP") == 0)
				{
					implementation_to_test = Ped::CUDA_HEATMAP;
					std::cout << "Implementation: CUDA_HEATMAP" << std::endl;
				}
				else {
					implementation_to_test = Ped::SEQ;
					std::cout << "Fallback implementation: SEQ" << std::endl;
				}
			}
			else
			{
				cerr << "Unrecognized command: \"" << argv[i] << "\". Ignoring ..." << endl;
			}
		}
		else // Assume it is a path to scenefile
		{
			scenefile = argv[i];
		}

		i += 1;
	}
	int retval = 0;
	{ // This scope is for the purpose of removing false memory leak positives

		// Reading the scenario file and setting up the crowd simulation model
		Ped::Model model;
		ParseScenario parser(scenefile);
		model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test, num_threads, split_factor, merge_factor);

		// Default number of steps to simulate. Feel free to change this.
		const int maxNumberOfStepsToSimulate = 10000;
		
				

		// Timing version
		// Run twice, without the gui, to compare the runtimes.
		// Compile with timing-release to enable this automatically.
		if (timing_mode)
		{
            //MainWindow mainwindow(model, timing_mode);
			// Run sequentially

			double fps_seq, fps_target;
			{
				Ped::Model model;
				ParseScenario parser(scenefile);
				model.setup(parser.getAgents(), parser.getWaypoints(), Ped::SEQ, 1, split_factor, merge_factor);
				PedSimulation simulation(model, NULL, timing_mode);
				// Simulation mode to use when profiling (without any GUI)
				std::cout << "Running reference version...\n";
				auto start = std::chrono::steady_clock::now();
				simulation.runSimulation(maxNumberOfStepsToSimulate);
				auto duration_seq = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
				fps_seq = ((float)simulation.getTickCount()) / ((float)duration_seq.count())*1000.0;
				cout << "Reference time: " << duration_seq.count() << " milliseconds, " << fps_seq << " Frames Per Second." << std::endl;
			}

			
			{
				Ped::Model model;
				ParseScenario parser(scenefile);
				model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test, num_threads, split_factor, merge_factor);
				PedSimulation simulation(model, NULL, timing_mode);
				// Simulation mode to use when profiling (without any GUI)
				std::cout << "Running target version...\n";
				auto start = std::chrono::steady_clock::now();
				simulation.runSimulation(maxNumberOfStepsToSimulate);
				auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
				fps_target = ((float)simulation.getTickCount()) / ((float)duration_target.count())*1000.0;
				cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;
			}
			std::cout << "\n\nSpeedup: " << fps_target / fps_seq << std::endl;
			
			

		}
		// Graphics version
		else
		{
            QApplication app(argc, argv);
            MainWindow mainwindow(model);

			PedSimulation simulation(model, &mainwindow, timing_mode);

			cout << "Demo setup complete, running ..." << endl;

			// Simulation mode to use when visualizing
			auto start = std::chrono::steady_clock::now();
			mainwindow.show();
			simulation.runSimulation(maxNumberOfStepsToSimulate);
			retval = app.exec();

			auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
			float fps = ((float)simulation.getTickCount()) / ((float)duration.count())*1000.0;
			cout << "Time: " << duration.count() << " milliseconds, " << fps << " Frames Per Second." << std::endl;
			
		}

		

		
	}

	cout << "Done" << endl;
	cout << "Type Enter to quit.." << endl;
	getchar(); // Wait for any key. Windows convenience...
	return retval;
}
