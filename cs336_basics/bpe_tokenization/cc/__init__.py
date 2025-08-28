import subprocess
import cppyy
import pathlib


RECOMPILE = False
CC_PATH = (pathlib.Path(__file__).resolve().parent)

cc_files = list(CC_PATH.glob("*.cc"))
cc_headers = list(CC_PATH.glob("*.h"))
cc_libs = list(CC_PATH.glob("*.so"))


def compile():
    for cc_file in cc_files:
        print(f"Compiling {cc_file}...")
        subprocess.run(
            [
                "clang++",
                "-std=c++20",
                "-Wall",
                "-fPIC",
                "-shared",
                "-O2",
                str(CC_PATH / cc_file),
                "-o",
                CC_PATH / f"lib{cc_file.stem}.so",
            ]
        )
    print("Done!")
    global cc_libs
    cc_libs = list(CC_PATH.glob("*.so"))


def load_libs():
    for header in cc_headers:
        cppyy.include(str(header))
    for lib in cc_libs:
        cppyy.load_library(str(lib))


if not cc_libs:
    compile()
load_libs()

from cppyy.gbl import bpe as cc  # type: ignore

BPEBuilder = cc.BPEBuilder  # type: ignore
cc.BPEBuilder.Train.__release_gil__ = True

TokenCollection = cc.TokenCollection  # type: ignore

