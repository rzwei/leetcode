# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2017.3.1\bin\cmake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2017.3.1\bin\cmake\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = F:\Users\rzhon\IdeaProjects\rzwei-leetcode\leetcode-vs\leetcode

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = F:\Users\rzhon\IdeaProjects\rzwei-leetcode\leetcode-vs\leetcode\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles\clion01.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\clion01.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\clion01.dir\flags.make

CMakeFiles\clion01.dir\leetcode2.cpp.obj: CMakeFiles\clion01.dir\flags.make
CMakeFiles\clion01.dir\leetcode2.cpp.obj: ..\leetcode2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=F:\Users\rzhon\IdeaProjects\rzwei-leetcode\leetcode-vs\leetcode\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/clion01.dir/leetcode2.cpp.obj"
	C:\PROGRA~2\MICROS~3\2017\COMMUN~1\VC\Tools\MSVC\1412~1.258\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\clion01.dir\leetcode2.cpp.obj /FdCMakeFiles\clion01.dir\ /FS -c F:\Users\rzhon\IdeaProjects\rzwei-leetcode\leetcode-vs\leetcode\leetcode2.cpp
<<

CMakeFiles\clion01.dir\leetcode2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/clion01.dir/leetcode2.cpp.i"
	C:\PROGRA~2\MICROS~3\2017\COMMUN~1\VC\Tools\MSVC\1412~1.258\bin\Hostx64\x64\cl.exe > CMakeFiles\clion01.dir\leetcode2.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E F:\Users\rzhon\IdeaProjects\rzwei-leetcode\leetcode-vs\leetcode\leetcode2.cpp
<<

CMakeFiles\clion01.dir\leetcode2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/clion01.dir/leetcode2.cpp.s"
	C:\PROGRA~2\MICROS~3\2017\COMMUN~1\VC\Tools\MSVC\1412~1.258\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\clion01.dir\leetcode2.cpp.s /c F:\Users\rzhon\IdeaProjects\rzwei-leetcode\leetcode-vs\leetcode\leetcode2.cpp
<<

CMakeFiles\clion01.dir\leetcode2.cpp.obj.requires:

.PHONY : CMakeFiles\clion01.dir\leetcode2.cpp.obj.requires

CMakeFiles\clion01.dir\leetcode2.cpp.obj.provides: CMakeFiles\clion01.dir\leetcode2.cpp.obj.requires
	$(MAKE) -f CMakeFiles\clion01.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\clion01.dir\leetcode2.cpp.obj.provides.build
.PHONY : CMakeFiles\clion01.dir\leetcode2.cpp.obj.provides

CMakeFiles\clion01.dir\leetcode2.cpp.obj.provides.build: CMakeFiles\clion01.dir\leetcode2.cpp.obj


# Object files for target clion01
clion01_OBJECTS = \
"CMakeFiles\clion01.dir\leetcode2.cpp.obj"

# External object files for target clion01
clion01_EXTERNAL_OBJECTS =

clion01.exe: CMakeFiles\clion01.dir\leetcode2.cpp.obj
clion01.exe: CMakeFiles\clion01.dir\build.make
clion01.exe: CMakeFiles\clion01.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=F:\Users\rzhon\IdeaProjects\rzwei-leetcode\leetcode-vs\leetcode\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable clion01.exe"
	"C:\Program Files\JetBrains\CLion 2017.3.1\bin\cmake\bin\cmake.exe" -E vs_link_exe --intdir=CMakeFiles\clion01.dir --manifests  -- C:\PROGRA~2\MICROS~3\2017\COMMUN~1\VC\Tools\MSVC\1412~1.258\bin\Hostx64\x64\link.exe /nologo @CMakeFiles\clion01.dir\objects1.rsp @<<
 /out:clion01.exe /implib:clion01.lib /pdb:F:\Users\rzhon\IdeaProjects\rzwei-leetcode\leetcode-vs\leetcode\cmake-build-debug\clion01.pdb /version:0.0  /machine:x64 /debug /INCREMENTAL /subsystem:console kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
<<

# Rule to build all files generated by this target.
CMakeFiles\clion01.dir\build: clion01.exe

.PHONY : CMakeFiles\clion01.dir\build

CMakeFiles\clion01.dir\requires: CMakeFiles\clion01.dir\leetcode2.cpp.obj.requires

.PHONY : CMakeFiles\clion01.dir\requires

CMakeFiles\clion01.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\clion01.dir\cmake_clean.cmake
.PHONY : CMakeFiles\clion01.dir\clean

CMakeFiles\clion01.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" F:\Users\rzhon\IdeaProjects\rzwei-leetcode\leetcode-vs\leetcode F:\Users\rzhon\IdeaProjects\rzwei-leetcode\leetcode-vs\leetcode F:\Users\rzhon\IdeaProjects\rzwei-leetcode\leetcode-vs\leetcode\cmake-build-debug F:\Users\rzhon\IdeaProjects\rzwei-leetcode\leetcode-vs\leetcode\cmake-build-debug F:\Users\rzhon\IdeaProjects\rzwei-leetcode\leetcode-vs\leetcode\cmake-build-debug\CMakeFiles\clion01.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\clion01.dir\depend

