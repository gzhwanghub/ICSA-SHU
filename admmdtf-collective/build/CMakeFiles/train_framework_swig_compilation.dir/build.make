# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mirror/wgz/admmdtf-collective

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mirror/wgz/admmdtf-collective/build

# Utility rule file for train_framework_swig_compilation.

# Include any custom commands dependencies for this target.
include CMakeFiles/train_framework_swig_compilation.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/train_framework_swig_compilation.dir/progress.make

CMakeFiles/train_framework_swig_compilation: CMakeFiles/train_framework.dir/train_frameworkPYTHON.stamp

CMakeFiles/train_framework.dir/train_frameworkPYTHON.stamp: ../src/train_framework.i
CMakeFiles/train_framework.dir/train_frameworkPYTHON.stamp: ../src/train_framework.i
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mirror/wgz/admmdtf-collective/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Swig compile src/train_framework.i for python"
	/usr/bin/cmake -E make_directory /mirror/wgz/admmdtf-collective/build/CMakeFiles/train_framework.dir /mirror/wgz/admmdtf-collective/build /mirror/wgz/admmdtf-collective/build/cpp
	/usr/bin/cmake -E touch /mirror/wgz/admmdtf-collective/build/CMakeFiles/train_framework.dir/train_frameworkPYTHON.stamp
	/usr/bin/cmake -E env SWIG_LIB=/usr/share/swig4.0 /usr/bin/swig4.0 -python -outdir /mirror/wgz/admmdtf-collective/build -c++ -interface _train_framework -I/usr/include/python3.10 -I/home/cluster/anaconda3/lib/python3.7/site-packages/mpi4py/include -I/mirror/wgz/admmdtf-collective/src -o /mirror/wgz/admmdtf-collective/build/cpp/train_frameworkPYTHON_wrap.cxx /mirror/wgz/admmdtf-collective/src/train_framework.i

train_framework_swig_compilation: CMakeFiles/train_framework.dir/train_frameworkPYTHON.stamp
train_framework_swig_compilation: CMakeFiles/train_framework_swig_compilation
train_framework_swig_compilation: CMakeFiles/train_framework_swig_compilation.dir/build.make
.PHONY : train_framework_swig_compilation

# Rule to build all files generated by this target.
CMakeFiles/train_framework_swig_compilation.dir/build: train_framework_swig_compilation
.PHONY : CMakeFiles/train_framework_swig_compilation.dir/build

CMakeFiles/train_framework_swig_compilation.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/train_framework_swig_compilation.dir/cmake_clean.cmake
.PHONY : CMakeFiles/train_framework_swig_compilation.dir/clean

CMakeFiles/train_framework_swig_compilation.dir/depend:
	cd /mirror/wgz/admmdtf-collective/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mirror/wgz/admmdtf-collective /mirror/wgz/admmdtf-collective /mirror/wgz/admmdtf-collective/build /mirror/wgz/admmdtf-collective/build /mirror/wgz/admmdtf-collective/build/CMakeFiles/train_framework_swig_compilation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/train_framework_swig_compilation.dir/depend

