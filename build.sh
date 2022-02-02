find ./ -maxdepth 1 -name "build" | xargs rm -rf
mkdir build && cd build
cmake -DBUILD_TPPS_INTREE=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$(dirname $(python3 -c 'import torch; print(torch.__file__)'));../cmake/Modules" -GNinja -DUSERCP=ON ..
ninja
cd ..
