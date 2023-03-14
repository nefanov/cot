import subprocess
import sys
def bench_uri_from_c_src(path: str):
    # clang -emit-llvm -c -O0 -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes myapp.c
    try:
        subprocess.run(["clang", "-emit-llvm", "-c", "-O0","-Xclang","-disable-O0-optnone", "-Xclang", "-disable-llvm-passes", path], capture_output=True)
    except Exception as e:
        print(e)
        sys.exit(1)
    ext = path.split(".")
    ext.pop()
    name = ".".join(ext) # required if there are many dots in the string
    return "file://" + ".".join([name, "bc"])
