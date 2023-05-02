clang -emit-llvm -c -O0 -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes $1 -o main.bc
opt -O2 main.bc -o opt_interm.bc
opt -dse -sroa -simplifycfg -jump-threading -deadargelim -simplifycfg -instcombine opt_interm.bc -o opt.bc
clang -c opt.bc
ls -ltr | tail -5
