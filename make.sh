#g++ -O0 -g -Wall -mavx512f -march=skylake-avx512 -std=c++17 main.cpp -lpthread -o logging_poc

#echo "g++ -g -Wall -mavx512f -mclwb -march=skylake-avx512 -std=c++17 main.cpp -lpthread -o logging_poc $@"
#g++ -g -Wall -mavx512f -mclwb -march=skylake-avx512 -std=c++17 main.cpp -lpthread -o logging_poc "$@"

echo "g++ -g -mclflushopt -Wall -std=c++17 main.cpp -lpthread -o logging_poc $@"
g++ -g -Wall -mclflushopt -std=c++17 main.cpp -lpthread -o logging_poc $@
