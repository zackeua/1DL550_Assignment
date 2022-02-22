Compile the code using:

$ make -j 4

Run the program using:

$ ./demo/demo [--help] [--timing-mode] [--threads X (number of threads)] [--split-factor X] [--merge-factor X] [-impl implementation (SEQ, OMP, PTHREAD, VECTOR, CUDA, SEQ1, SEQ2, MOVE_SEQ, MOVE_CONSTANT, MOVE_ADAPTIVE)] [scenario]

Option 1: Use --threads to specify the number of threads (defaults to 1).
Option 2: Use --impl to specify implementation (defaults to SEQ).
Option 3: Use --split-factor to tune the threshold for when the regions split.
Option 4: Use --merge-factor to tune the threshold for when the regions merge.