find ./ -maxdepth 1 -name "build" | xargs rm -rf
mkdir build && cd build
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-g" -DCMAKE_PREFIX_PATH="$(dirname $(python3 -c 'import torch; print(torch.__file__)'));../cmake/Modules" -GNinja -DUSERCP=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx ..
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(dirname $(python3 -c 'import torch; print(torch.__file__)'));../cmake/Modules" -GNinja -DUSERCP=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx ..
ninja
cd ..