# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.4

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/drinkingcoder/Documents/Code/KinectFusion

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/drinkingcoder/Documents/Code/KinectFusion/build

# Include any dependencies generated for this target.
include src/CMakeFiles/testconfig.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/testconfig.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/testconfig.dir/flags.make

src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.o: src/CMakeFiles/testconfig.dir/flags.make
src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.o: ../src/TestConfigurator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/drinkingcoder/Documents/Code/KinectFusion/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.o"
	cd /home/drinkingcoder/Documents/Code/KinectFusion/build/src && g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testconfig.dir/TestConfigurator.cpp.o -c /home/drinkingcoder/Documents/Code/KinectFusion/src/TestConfigurator.cpp

src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testconfig.dir/TestConfigurator.cpp.i"
	cd /home/drinkingcoder/Documents/Code/KinectFusion/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/drinkingcoder/Documents/Code/KinectFusion/src/TestConfigurator.cpp > CMakeFiles/testconfig.dir/TestConfigurator.cpp.i

src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testconfig.dir/TestConfigurator.cpp.s"
	cd /home/drinkingcoder/Documents/Code/KinectFusion/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/drinkingcoder/Documents/Code/KinectFusion/src/TestConfigurator.cpp -o CMakeFiles/testconfig.dir/TestConfigurator.cpp.s

src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.o.requires:

.PHONY : src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.o.requires

src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.o.provides: src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/testconfig.dir/build.make src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.o.provides.build
.PHONY : src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.o.provides

src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.o.provides.build: src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.o


# Object files for target testconfig
testconfig_OBJECTS = \
"CMakeFiles/testconfig.dir/TestConfigurator.cpp.o"

# External object files for target testconfig
testconfig_EXTERNAL_OBJECTS =

../bin/testconfig: src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.o
../bin/testconfig: src/CMakeFiles/testconfig.dir/build.make
../bin/testconfig: src/CMakeFiles/testconfig.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/drinkingcoder/Documents/Code/KinectFusion/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/testconfig"
	cd /home/drinkingcoder/Documents/Code/KinectFusion/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testconfig.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/testconfig.dir/build: ../bin/testconfig

.PHONY : src/CMakeFiles/testconfig.dir/build

src/CMakeFiles/testconfig.dir/requires: src/CMakeFiles/testconfig.dir/TestConfigurator.cpp.o.requires

.PHONY : src/CMakeFiles/testconfig.dir/requires

src/CMakeFiles/testconfig.dir/clean:
	cd /home/drinkingcoder/Documents/Code/KinectFusion/build/src && $(CMAKE_COMMAND) -P CMakeFiles/testconfig.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/testconfig.dir/clean

src/CMakeFiles/testconfig.dir/depend:
	cd /home/drinkingcoder/Documents/Code/KinectFusion/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/drinkingcoder/Documents/Code/KinectFusion /home/drinkingcoder/Documents/Code/KinectFusion/src /home/drinkingcoder/Documents/Code/KinectFusion/build /home/drinkingcoder/Documents/Code/KinectFusion/build/src /home/drinkingcoder/Documents/Code/KinectFusion/build/src/CMakeFiles/testconfig.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/testconfig.dir/depend
