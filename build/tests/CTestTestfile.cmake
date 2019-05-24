# CMake generated Testfile for 
# Source directory: /Users/kabicm/Projects/tiled-mm/tests
# Build directory: /Users/kabicm/Projects/tiled-mm/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(square-small "/Users/kabicm/Projects/tiled-mm/build/tests/test-multiply" "-m" "1000" "-n" "1000" "-k" "1000")
add_test(square-large "/Users/kabicm/Projects/tiled-mm/build/tests/test-multiply" "-m" "10000" "-n" "10000" "-k" "10000")
add_test(non-square-small "/Users/kabicm/Projects/tiled-mm/build/tests/test-multiply" "-m" "1234" "-n" "4567" "-k" "1357")
add_test(non-square-large "/Users/kabicm/Projects/tiled-mm/build/tests/test-multiply" "-m" "12345" "-n" "23456" "-k" "67891")
