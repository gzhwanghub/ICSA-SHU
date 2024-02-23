# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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
CMAKE_COMMAND = /opt/clion/bin/cmake/linux/x64/bin/cmake

# The command to remove a file.
RM = /opt/clion/bin/cmake/linux/x64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mirror/wgz/admm_collective

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mirror/wgz/admm_collective/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/admm_collective.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/admm_collective.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/admm_collective.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/admm_collective.dir/flags.make

CMakeFiles/admm_collective.dir/src/collective.cpp.o: CMakeFiles/admm_collective.dir/flags.make
CMakeFiles/admm_collective.dir/src/collective.cpp.o: /mirror/wgz/admm_collective/src/collective.cpp
CMakeFiles/admm_collective.dir/src/collective.cpp.o: CMakeFiles/admm_collective.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/admm_collective.dir/src/collective.cpp.o"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/admm_collective.dir/src/collective.cpp.o -MF CMakeFiles/admm_collective.dir/src/collective.cpp.o.d -o CMakeFiles/admm_collective.dir/src/collective.cpp.o -c /mirror/wgz/admm_collective/src/collective.cpp

CMakeFiles/admm_collective.dir/src/collective.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/admm_collective.dir/src/collective.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mirror/wgz/admm_collective/src/collective.cpp > CMakeFiles/admm_collective.dir/src/collective.cpp.i

CMakeFiles/admm_collective.dir/src/collective.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/admm_collective.dir/src/collective.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mirror/wgz/admm_collective/src/collective.cpp -o CMakeFiles/admm_collective.dir/src/collective.cpp.s

CMakeFiles/admm_collective.dir/src/gd.cpp.o: CMakeFiles/admm_collective.dir/flags.make
CMakeFiles/admm_collective.dir/src/gd.cpp.o: /mirror/wgz/admm_collective/src/gd.cpp
CMakeFiles/admm_collective.dir/src/gd.cpp.o: CMakeFiles/admm_collective.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/admm_collective.dir/src/gd.cpp.o"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/admm_collective.dir/src/gd.cpp.o -MF CMakeFiles/admm_collective.dir/src/gd.cpp.o.d -o CMakeFiles/admm_collective.dir/src/gd.cpp.o -c /mirror/wgz/admm_collective/src/gd.cpp

CMakeFiles/admm_collective.dir/src/gd.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/admm_collective.dir/src/gd.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mirror/wgz/admm_collective/src/gd.cpp > CMakeFiles/admm_collective.dir/src/gd.cpp.i

CMakeFiles/admm_collective.dir/src/gd.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/admm_collective.dir/src/gd.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mirror/wgz/admm_collective/src/gd.cpp -o CMakeFiles/admm_collective.dir/src/gd.cpp.s

CMakeFiles/admm_collective.dir/src/lbfgs.cpp.o: CMakeFiles/admm_collective.dir/flags.make
CMakeFiles/admm_collective.dir/src/lbfgs.cpp.o: /mirror/wgz/admm_collective/src/lbfgs.cpp
CMakeFiles/admm_collective.dir/src/lbfgs.cpp.o: CMakeFiles/admm_collective.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/admm_collective.dir/src/lbfgs.cpp.o"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/admm_collective.dir/src/lbfgs.cpp.o -MF CMakeFiles/admm_collective.dir/src/lbfgs.cpp.o.d -o CMakeFiles/admm_collective.dir/src/lbfgs.cpp.o -c /mirror/wgz/admm_collective/src/lbfgs.cpp

CMakeFiles/admm_collective.dir/src/lbfgs.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/admm_collective.dir/src/lbfgs.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mirror/wgz/admm_collective/src/lbfgs.cpp > CMakeFiles/admm_collective.dir/src/lbfgs.cpp.i

CMakeFiles/admm_collective.dir/src/lbfgs.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/admm_collective.dir/src/lbfgs.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mirror/wgz/admm_collective/src/lbfgs.cpp -o CMakeFiles/admm_collective.dir/src/lbfgs.cpp.s

CMakeFiles/admm_collective.dir/src/logistic.cpp.o: CMakeFiles/admm_collective.dir/flags.make
CMakeFiles/admm_collective.dir/src/logistic.cpp.o: /mirror/wgz/admm_collective/src/logistic.cpp
CMakeFiles/admm_collective.dir/src/logistic.cpp.o: CMakeFiles/admm_collective.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/admm_collective.dir/src/logistic.cpp.o"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/admm_collective.dir/src/logistic.cpp.o -MF CMakeFiles/admm_collective.dir/src/logistic.cpp.o.d -o CMakeFiles/admm_collective.dir/src/logistic.cpp.o -c /mirror/wgz/admm_collective/src/logistic.cpp

CMakeFiles/admm_collective.dir/src/logistic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/admm_collective.dir/src/logistic.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mirror/wgz/admm_collective/src/logistic.cpp > CMakeFiles/admm_collective.dir/src/logistic.cpp.i

CMakeFiles/admm_collective.dir/src/logistic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/admm_collective.dir/src/logistic.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mirror/wgz/admm_collective/src/logistic.cpp -o CMakeFiles/admm_collective.dir/src/logistic.cpp.s

CMakeFiles/admm_collective.dir/src/math_util.cpp.o: CMakeFiles/admm_collective.dir/flags.make
CMakeFiles/admm_collective.dir/src/math_util.cpp.o: /mirror/wgz/admm_collective/src/math_util.cpp
CMakeFiles/admm_collective.dir/src/math_util.cpp.o: CMakeFiles/admm_collective.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/admm_collective.dir/src/math_util.cpp.o"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/admm_collective.dir/src/math_util.cpp.o -MF CMakeFiles/admm_collective.dir/src/math_util.cpp.o.d -o CMakeFiles/admm_collective.dir/src/math_util.cpp.o -c /mirror/wgz/admm_collective/src/math_util.cpp

CMakeFiles/admm_collective.dir/src/math_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/admm_collective.dir/src/math_util.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mirror/wgz/admm_collective/src/math_util.cpp > CMakeFiles/admm_collective.dir/src/math_util.cpp.i

CMakeFiles/admm_collective.dir/src/math_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/admm_collective.dir/src/math_util.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mirror/wgz/admm_collective/src/math_util.cpp -o CMakeFiles/admm_collective.dir/src/math_util.cpp.s

CMakeFiles/admm_collective.dir/src/prob.cpp.o: CMakeFiles/admm_collective.dir/flags.make
CMakeFiles/admm_collective.dir/src/prob.cpp.o: /mirror/wgz/admm_collective/src/prob.cpp
CMakeFiles/admm_collective.dir/src/prob.cpp.o: CMakeFiles/admm_collective.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/admm_collective.dir/src/prob.cpp.o"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/admm_collective.dir/src/prob.cpp.o -MF CMakeFiles/admm_collective.dir/src/prob.cpp.o.d -o CMakeFiles/admm_collective.dir/src/prob.cpp.o -c /mirror/wgz/admm_collective/src/prob.cpp

CMakeFiles/admm_collective.dir/src/prob.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/admm_collective.dir/src/prob.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mirror/wgz/admm_collective/src/prob.cpp > CMakeFiles/admm_collective.dir/src/prob.cpp.i

CMakeFiles/admm_collective.dir/src/prob.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/admm_collective.dir/src/prob.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mirror/wgz/admm_collective/src/prob.cpp -o CMakeFiles/admm_collective.dir/src/prob.cpp.s

CMakeFiles/admm_collective.dir/src/properties.cpp.o: CMakeFiles/admm_collective.dir/flags.make
CMakeFiles/admm_collective.dir/src/properties.cpp.o: /mirror/wgz/admm_collective/src/properties.cpp
CMakeFiles/admm_collective.dir/src/properties.cpp.o: CMakeFiles/admm_collective.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/admm_collective.dir/src/properties.cpp.o"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/admm_collective.dir/src/properties.cpp.o -MF CMakeFiles/admm_collective.dir/src/properties.cpp.o.d -o CMakeFiles/admm_collective.dir/src/properties.cpp.o -c /mirror/wgz/admm_collective/src/properties.cpp

CMakeFiles/admm_collective.dir/src/properties.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/admm_collective.dir/src/properties.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mirror/wgz/admm_collective/src/properties.cpp > CMakeFiles/admm_collective.dir/src/properties.cpp.i

CMakeFiles/admm_collective.dir/src/properties.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/admm_collective.dir/src/properties.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mirror/wgz/admm_collective/src/properties.cpp -o CMakeFiles/admm_collective.dir/src/properties.cpp.s

CMakeFiles/admm_collective.dir/src/string_util.cpp.o: CMakeFiles/admm_collective.dir/flags.make
CMakeFiles/admm_collective.dir/src/string_util.cpp.o: /mirror/wgz/admm_collective/src/string_util.cpp
CMakeFiles/admm_collective.dir/src/string_util.cpp.o: CMakeFiles/admm_collective.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/admm_collective.dir/src/string_util.cpp.o"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/admm_collective.dir/src/string_util.cpp.o -MF CMakeFiles/admm_collective.dir/src/string_util.cpp.o.d -o CMakeFiles/admm_collective.dir/src/string_util.cpp.o -c /mirror/wgz/admm_collective/src/string_util.cpp

CMakeFiles/admm_collective.dir/src/string_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/admm_collective.dir/src/string_util.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mirror/wgz/admm_collective/src/string_util.cpp > CMakeFiles/admm_collective.dir/src/string_util.cpp.i

CMakeFiles/admm_collective.dir/src/string_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/admm_collective.dir/src/string_util.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mirror/wgz/admm_collective/src/string_util.cpp -o CMakeFiles/admm_collective.dir/src/string_util.cpp.s

CMakeFiles/admm_collective.dir/src/svm.cpp.o: CMakeFiles/admm_collective.dir/flags.make
CMakeFiles/admm_collective.dir/src/svm.cpp.o: /mirror/wgz/admm_collective/src/svm.cpp
CMakeFiles/admm_collective.dir/src/svm.cpp.o: CMakeFiles/admm_collective.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/admm_collective.dir/src/svm.cpp.o"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/admm_collective.dir/src/svm.cpp.o -MF CMakeFiles/admm_collective.dir/src/svm.cpp.o.d -o CMakeFiles/admm_collective.dir/src/svm.cpp.o -c /mirror/wgz/admm_collective/src/svm.cpp

CMakeFiles/admm_collective.dir/src/svm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/admm_collective.dir/src/svm.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mirror/wgz/admm_collective/src/svm.cpp > CMakeFiles/admm_collective.dir/src/svm.cpp.i

CMakeFiles/admm_collective.dir/src/svm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/admm_collective.dir/src/svm.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mirror/wgz/admm_collective/src/svm.cpp -o CMakeFiles/admm_collective.dir/src/svm.cpp.s

CMakeFiles/admm_collective.dir/src/sync_admm.cpp.o: CMakeFiles/admm_collective.dir/flags.make
CMakeFiles/admm_collective.dir/src/sync_admm.cpp.o: /mirror/wgz/admm_collective/src/sync_admm.cpp
CMakeFiles/admm_collective.dir/src/sync_admm.cpp.o: CMakeFiles/admm_collective.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/admm_collective.dir/src/sync_admm.cpp.o"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/admm_collective.dir/src/sync_admm.cpp.o -MF CMakeFiles/admm_collective.dir/src/sync_admm.cpp.o.d -o CMakeFiles/admm_collective.dir/src/sync_admm.cpp.o -c /mirror/wgz/admm_collective/src/sync_admm.cpp

CMakeFiles/admm_collective.dir/src/sync_admm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/admm_collective.dir/src/sync_admm.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mirror/wgz/admm_collective/src/sync_admm.cpp > CMakeFiles/admm_collective.dir/src/sync_admm.cpp.i

CMakeFiles/admm_collective.dir/src/sync_admm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/admm_collective.dir/src/sync_admm.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mirror/wgz/admm_collective/src/sync_admm.cpp -o CMakeFiles/admm_collective.dir/src/sync_admm.cpp.s

CMakeFiles/admm_collective.dir/src/train.cpp.o: CMakeFiles/admm_collective.dir/flags.make
CMakeFiles/admm_collective.dir/src/train.cpp.o: /mirror/wgz/admm_collective/src/train.cpp
CMakeFiles/admm_collective.dir/src/train.cpp.o: CMakeFiles/admm_collective.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/admm_collective.dir/src/train.cpp.o"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/admm_collective.dir/src/train.cpp.o -MF CMakeFiles/admm_collective.dir/src/train.cpp.o.d -o CMakeFiles/admm_collective.dir/src/train.cpp.o -c /mirror/wgz/admm_collective/src/train.cpp

CMakeFiles/admm_collective.dir/src/train.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/admm_collective.dir/src/train.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mirror/wgz/admm_collective/src/train.cpp > CMakeFiles/admm_collective.dir/src/train.cpp.i

CMakeFiles/admm_collective.dir/src/train.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/admm_collective.dir/src/train.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mirror/wgz/admm_collective/src/train.cpp -o CMakeFiles/admm_collective.dir/src/train.cpp.s

CMakeFiles/admm_collective.dir/src/tron.cpp.o: CMakeFiles/admm_collective.dir/flags.make
CMakeFiles/admm_collective.dir/src/tron.cpp.o: /mirror/wgz/admm_collective/src/tron.cpp
CMakeFiles/admm_collective.dir/src/tron.cpp.o: CMakeFiles/admm_collective.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/admm_collective.dir/src/tron.cpp.o"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/admm_collective.dir/src/tron.cpp.o -MF CMakeFiles/admm_collective.dir/src/tron.cpp.o.d -o CMakeFiles/admm_collective.dir/src/tron.cpp.o -c /mirror/wgz/admm_collective/src/tron.cpp

CMakeFiles/admm_collective.dir/src/tron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/admm_collective.dir/src/tron.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mirror/wgz/admm_collective/src/tron.cpp > CMakeFiles/admm_collective.dir/src/tron.cpp.i

CMakeFiles/admm_collective.dir/src/tron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/admm_collective.dir/src/tron.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mirror/wgz/admm_collective/src/tron.cpp -o CMakeFiles/admm_collective.dir/src/tron.cpp.s

# Object files for target admm_collective
admm_collective_OBJECTS = \
"CMakeFiles/admm_collective.dir/src/collective.cpp.o" \
"CMakeFiles/admm_collective.dir/src/gd.cpp.o" \
"CMakeFiles/admm_collective.dir/src/lbfgs.cpp.o" \
"CMakeFiles/admm_collective.dir/src/logistic.cpp.o" \
"CMakeFiles/admm_collective.dir/src/math_util.cpp.o" \
"CMakeFiles/admm_collective.dir/src/prob.cpp.o" \
"CMakeFiles/admm_collective.dir/src/properties.cpp.o" \
"CMakeFiles/admm_collective.dir/src/string_util.cpp.o" \
"CMakeFiles/admm_collective.dir/src/svm.cpp.o" \
"CMakeFiles/admm_collective.dir/src/sync_admm.cpp.o" \
"CMakeFiles/admm_collective.dir/src/train.cpp.o" \
"CMakeFiles/admm_collective.dir/src/tron.cpp.o"

# External object files for target admm_collective
admm_collective_EXTERNAL_OBJECTS =

/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/src/collective.cpp.o
/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/src/gd.cpp.o
/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/src/lbfgs.cpp.o
/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/src/logistic.cpp.o
/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/src/math_util.cpp.o
/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/src/prob.cpp.o
/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/src/properties.cpp.o
/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/src/string_util.cpp.o
/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/src/svm.cpp.o
/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/src/sync_admm.cpp.o
/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/src/train.cpp.o
/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/src/tron.cpp.o
/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/build.make
/mirror/wgz/admm_collective/bin/admm_collective: CMakeFiles/admm_collective.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Linking CXX executable /mirror/wgz/admm_collective/bin/admm_collective"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/admm_collective.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/admm_collective.dir/build: /mirror/wgz/admm_collective/bin/admm_collective
.PHONY : CMakeFiles/admm_collective.dir/build

CMakeFiles/admm_collective.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/admm_collective.dir/cmake_clean.cmake
.PHONY : CMakeFiles/admm_collective.dir/clean

CMakeFiles/admm_collective.dir/depend:
	cd /mirror/wgz/admm_collective/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mirror/wgz/admm_collective /mirror/wgz/admm_collective /mirror/wgz/admm_collective/cmake-build-debug /mirror/wgz/admm_collective/cmake-build-debug /mirror/wgz/admm_collective/cmake-build-debug/CMakeFiles/admm_collective.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/admm_collective.dir/depend

