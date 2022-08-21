# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

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

# Produce verbose output by default.
VERBOSE = 1

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
CMAKE_SOURCE_DIR = /home/ashah122/taskflow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ashah122/taskflow/build

# Include any dependencies generated for this target.
include benchmarks/CMakeFiles/hetero_traversal.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include benchmarks/CMakeFiles/hetero_traversal.dir/compiler_depend.make

# Include the progress variables for this target.
include benchmarks/CMakeFiles/hetero_traversal.dir/progress.make

# Include the compile flags for this target's objects.
include benchmarks/CMakeFiles/hetero_traversal.dir/flags.make

benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.o: benchmarks/CMakeFiles/hetero_traversal.dir/flags.make
benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.o: ../benchmarks/hetero_traversal/main.cpp
benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.o: benchmarks/CMakeFiles/hetero_traversal.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ashah122/taskflow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.o"
	cd /home/ashah122/taskflow/build/benchmarks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.o -MF CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.o.d -o CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.o -c /home/ashah122/taskflow/benchmarks/hetero_traversal/main.cpp

benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.i"
	cd /home/ashah122/taskflow/build/benchmarks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ashah122/taskflow/benchmarks/hetero_traversal/main.cpp > CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.i

benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.s"
	cd /home/ashah122/taskflow/build/benchmarks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ashah122/taskflow/benchmarks/hetero_traversal/main.cpp -o CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.s

benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.o: benchmarks/CMakeFiles/hetero_traversal.dir/flags.make
benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.o: ../benchmarks/hetero_traversal/taskflow.cpp
benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.o: benchmarks/CMakeFiles/hetero_traversal.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ashah122/taskflow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.o"
	cd /home/ashah122/taskflow/build/benchmarks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.o -MF CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.o.d -o CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.o -c /home/ashah122/taskflow/benchmarks/hetero_traversal/taskflow.cpp

benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.i"
	cd /home/ashah122/taskflow/build/benchmarks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ashah122/taskflow/benchmarks/hetero_traversal/taskflow.cpp > CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.i

benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.s"
	cd /home/ashah122/taskflow/build/benchmarks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ashah122/taskflow/benchmarks/hetero_traversal/taskflow.cpp -o CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.s

# Object files for target hetero_traversal
hetero_traversal_OBJECTS = \
"CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.o" \
"CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.o"

# External object files for target hetero_traversal
hetero_traversal_EXTERNAL_OBJECTS =

benchmarks/hetero_traversal: benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/main.cpp.o
benchmarks/hetero_traversal: benchmarks/CMakeFiles/hetero_traversal.dir/hetero_traversal/taskflow.cpp.o
benchmarks/hetero_traversal: benchmarks/CMakeFiles/hetero_traversal.dir/build.make
benchmarks/hetero_traversal: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
benchmarks/hetero_traversal: /usr/lib/x86_64-linux-gnu/libpthread.so
benchmarks/hetero_traversal: benchmarks/CMakeFiles/hetero_traversal.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ashah122/taskflow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable hetero_traversal"
	cd /home/ashah122/taskflow/build/benchmarks && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hetero_traversal.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
benchmarks/CMakeFiles/hetero_traversal.dir/build: benchmarks/hetero_traversal
.PHONY : benchmarks/CMakeFiles/hetero_traversal.dir/build

benchmarks/CMakeFiles/hetero_traversal.dir/clean:
	cd /home/ashah122/taskflow/build/benchmarks && $(CMAKE_COMMAND) -P CMakeFiles/hetero_traversal.dir/cmake_clean.cmake
.PHONY : benchmarks/CMakeFiles/hetero_traversal.dir/clean

benchmarks/CMakeFiles/hetero_traversal.dir/depend:
	cd /home/ashah122/taskflow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ashah122/taskflow /home/ashah122/taskflow/benchmarks /home/ashah122/taskflow/build /home/ashah122/taskflow/build/benchmarks /home/ashah122/taskflow/build/benchmarks/CMakeFiles/hetero_traversal.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : benchmarks/CMakeFiles/hetero_traversal.dir/depend
