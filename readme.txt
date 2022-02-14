Compile the code using:

$ make -j 4

Run the program using:

$ ./demo/demo [--help] [--timing-mode] [--threads X (number of threads)] [--impl implementation (SEQ, OMP, PTHREAD, VECTOR, CUDA, SEQ1, SEQ2)] [scenario]

Optional 1: Use --threads to specify the number of threads (defaults to 1)
Optional 2: Use --impl to specify implementation (dafaults to SEQ)
