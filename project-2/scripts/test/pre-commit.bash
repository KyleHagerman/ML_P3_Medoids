#!/usr/bin/env bash

# magic line to ensure that we're always inside the root of our application,
# no matter from which directory we'll run script
# thanks to it we can just enter `./scripts/run-tests.bash`
cd "${0%/*}/.."

echo "Running pre-commit hook"
./run-tests.bash

# $? stores exit value of the last command
if [ $? -ne 0 ]; then
 echo "Tests must pass before commit!"
 exit 1
fi
