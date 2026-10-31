# SPDX-License-Identifier: GPL-3.0-or-later
"""Build actual ORB-SLAM3/DBoW2/g2o with a headless display-only patch.

Run inside Linux/WSL. --source is the pinned original checkout, --prefix holds
private OpenCV/Eigen/Boost/OpenSSL packages. Outputs stay in --output. Compilation
uses one/two jobs (Optimizer exclusive) and Eigen parallelism is disabled. No global installation is used.
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def prepare(source, patched):
    def write(path, content):
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
    if not patched.exists():
        for name in ("src", "include", "Thirdparty"):
            shutil.copytree(source / name, patched / name, dirs_exist_ok=True)
    for path in [*patched.rglob("*.h"), *patched.rglob("*.cc")]:
        content = path.read_text(encoding="utf-8")
        content = re.sub(r"#include\s*[<\"]pangolin/pangolin.h[>\"]", "", content)
        content = re.sub(r"#include\s*[<\"]opencv2/highgui(?:/highgui)?\.hpp[>\"]", "", content)
        content = content.replace("GLubyte", "unsigned char")
        content = content.replace("#include <opencv2/opencv.hpp>", "#include <opencv2/core.hpp>\n#include <opencv2/imgproc.hpp>\n#include <opencv2/features2d.hpp>\n#include <opencv2/calib3d.hpp>")
        write(path, content)
    for name, definition in {
        "MapDrawer.h": '''#include "Atlas.h"
namespace ORB_SLAM3 { class Settings; class MapDrawer {
public: MapDrawer(Atlas*, const std::string&, Settings*) {}
void SetCurrentCameraPose(const Sophus::SE3f&) {}
}; }
''',
        "Viewer.h": '''#include <stdexcept>
namespace ORB_SLAM3 { class System; class FrameDrawer; class MapDrawer; class Tracking; class Settings;
class Viewer { public:
Viewer(System*,FrameDrawer*,MapDrawer*,Tracking*,const std::string&,Settings*) { throw std::runtime_error("Headless build requires bUseViewer=false"); }
void Run() {} void RequestStop() {} bool isStopped() {return true;} void Release() {}
void RequestFinish() {} bool isFinished() {return true;} bool both=false;
}; }
''',
    }.items():
        target = patched / "include" / name
        original = (source / "include" / name).read_text(encoding="utf-8")
        license_header = original[:original.index("#ifndef")]
        write(target, license_header+"#pragma once\n"+definition)
    system = patched / "src/System.cc"
    content = system.read_text(encoding="utf-8")
    if "mptLocalMapping->joinable()" not in content:
        content = content.replace("mpLoopCloser->RequestFinish();", "mpLoopCloser->RequestFinish();\n    if(mptLocalMapping->joinable()) mptLocalMapping->join();\n    if(mptLoopClosing->joinable()) mptLoopClosing->join();")
    write(system, content)
    config = patched / "Thirdparty/g2o/config.h"
    write(config, "#pragma once\n/* Static, sequential native build: no OpenMP or shared export macros. */\n")


def build(source, prefix, output, cache=None, headers=None, jobs=1):
    source, prefix, output = map(lambda p: Path(p).resolve(), (source, prefix, output))
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache).resolve() if cache else output
    patched = cache / "source"
    prepare(source, patched)
    local_headers = cache / "headers"
    if not local_headers.exists():
        for name in ("opencv4", "openssl", "x86_64-linux-gnu/openssl"):
            source_path = prefix / "usr/include" / name
            if source_path.exists():
                shutil.copytree(source_path, local_headers / name, dirs_exist_ok=True)
    shared = Path(headers).resolve() if headers else prefix / "usr/include"
    includes = [patched, patched / "include", patched / "include/CameraModels", patched / "Thirdparty/Sophus",
                local_headers, local_headers / "opencv4", local_headers / "x86_64-linux-gnu", shared, shared / "eigen3"]
    flags = ["-std=c++14", "-O0", "-g0", "-DNDEBUG", "-DCOMPILEDWITHC11", "-DEIGEN_DONT_PARALLELIZE", "-pthread"]
    flags += ["-I"+str(path) for path in includes]
    cmake = (source / "CMakeLists.txt").read_text(encoding="utf-8")
    main = re.search(r"add_library\(.*?SHARED(.*?)\)\s*add_subdirectory", cmake, re.S | re.I).group(1)
    files = [patched / name for name in re.findall(r"src/[^\s]+\.(?:cc|cpp)", main) if not name.endswith(("MapDrawer.cc", "Viewer.cc"))]
    files += list((patched / "Thirdparty/DBoW2/DBoW2").glob("*.cpp"))+list((patched / "Thirdparty/DBoW2/DUtils").glob("*.cpp"))
    g2o = (source / "Thirdparty/g2o/CMakeLists.txt").read_text(encoding="utf-8")
    files += [patched / "Thirdparty/g2o" / name for name in re.findall(r"g2o/[^\s]+\.(?:cpp|c)(?=\s)", g2o)]
    files += [Path(__file__).with_name("runner.cpp").resolve()]
    object_directory = cache / "objects"
    object_directory.mkdir(parents=True, exist_ok=True)
    objects = [str(object_directory / f"unit-{index:03d}.o") for index in range(len(files))]

    def compile_unit(item):
        index, path = item
        target = object_directory / f"unit-{index:03d}.o"
        dependency = target.with_suffix(".d")
        if target.exists() and target.stat().st_size > 0 and dependency.exists():
            text = dependency.read_text().replace("\\\n", "")
            dependencies = [Path(name) for name in shlex.split(text.split(":", 1)[1])]
            if all(name.exists() and name.stat().st_mtime <= target.stat().st_mtime for name in dependencies):
                if target.parent != output:
                    shutil.copy2(target, output / target.name)
                    shutil.copy2(dependency, output / dependency.name)
                return
        compiler = "gcc" if path.suffix == ".c" else "g++"
        command = [compiler]+([x for x in flags if not x.startswith("-std=")] if compiler == "gcc" else flags)
        temporary = target.with_suffix(".partial.o")
        command += ["-MMD", "-MF", str(dependency), "-c", str(path), "-o", str(temporary)]
        print(f"[{index+1}/{len(files)}] {path.name}", flush=True)
        subprocess.run(command, check=True)
        temporary.replace(target)
        if target.parent != output:
            shutil.copy2(target, output / target.name)
            shutil.copy2(dependency, output / dependency.name)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pending = []
        for item in enumerate(files):
            if item[1].name == "Optimizer.cc":
                for future in pending:
                    future.result()
                pending = []
                compile_unit(item)
            else:
                pending.append(executor.submit(compile_unit, item))
        for future in pending:
            future.result()
    libraries = prefix / "usr/lib/x86_64-linux-gnu"
    names = ["opencv_core", "opencv_imgproc", "opencv_features2d", "opencv_calib3d", "opencv_flann", "boost_serialization", ":libcrypto.so.3"]
    command = ["g++", "-pthread", *objects, "-L"+str(libraries), "-Wl,-rpath,"+str(libraries), "-Wl,-rpath-link,"+str(libraries)]
    executable = cache / "orbslam3_runner"
    command += ["-l"+name for name in names]+["-o", str(executable)]
    environment = dict(os.environ, LD_LIBRARY_PATH=str(libraries)+":"+str(libraries / "openblas-pthread"))
    subprocess.run(command, check=True, env=environment)
    if executable.parent != output:
        shutil.copy2(executable, output / "orbslam3_runner")
    print(output / "orbslam3_runner", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "prefix", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--cache", help="Optional Linux-native source/header cache (avoids /mnt/d metadata latency)")
    parser.add_argument("--headers", help="Optional read-only Boost/Eigen include prefix")
    parser.add_argument("--jobs", type=int, choices=[1, 2], default=1, help="Independent compiler jobs; Optimizer.cc remains exclusive")
    args = parser.parse_args()
    build(args.source, args.prefix, args.output, args.cache, args.headers, args.jobs)
