# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# this is just a bypass for this module name collision with built-in one

test_parallel_jobs="${NUM_PARALLEL_TESTS:-5}"
# this is the directory where the tests are located
test_dirs=$1 # parse the first argument
printf "Running tests in '$test_dirs'\n"
test_args=$2 # parse the first argument
printf "Running tests with args '$test_args'\n"
COLLECTED_TESTS_FILE="collected_tests.txt"

ls -lh .  # show the contents of the directory

# python arguments
defaults=" -m pytest -v --cov=torchmetrics --durations=50 " + $test_args
echo "Using defaults: ${defaults}"

# get the list of parametrizations. we need to call them separately. the last two lines are removed.
# note: if there's a syntax error, this will fail with some garbled output
python -um pytest -q $test_dirs -m DDP --collect-only --pythonwarnings ignore 2>&1 > $COLLECTED_TESTS_FILE
# early terminate if collection failed (e.g. syntax error)
if [[ $? != 0 ]]; then
  cat $COLLECTED_TESTS_FILE
  exit 1
fi

# removes the last line of the file
sed -i '$d' $COLLECTED_TESTS_FILE

# Get test list and run each test individually
tests=($(grep -oP '\S+::test_\S+' "$COLLECTED_TESTS_FILE"))
test_count=${#tests[@]}
# present the collected tests
printf "collected $test_count tests:\n-------------------\n"
echo $(IFS='\n'; echo "${tests[@]}")
printf "\n===================\n"

# if test count is one print warning
if [[ $test_count -eq 1 ]]; then
  printf "WARNING: only one test found!\n"
elif [ $test_count -eq 0 ]; then
  printf "ERROR: no tests found!\n"
  exit 1
fi

# clear all the collected reports
rm -f parallel_test_output-*.txt  # in case it exists, remove it

# round up the number of tests to the next integer
test_batch_size=$(((test_count + test_parallel_jobs - 1) / test_parallel_jobs))
printf "Running $test_count tests in $test_parallel_jobs jobs batched by $test_batch_size\n"
pids=() # array of PID for running tests
for i in {0..$test_parallel_jobs}; do
  begin=$((i*test_batch_size))
  end=$((begin+test_batch_size))
  tests_batch=${tests[@]:$begin:$end}
  printf "Batch $i with ${#tests_batch[@]} tests"
  tests_batch=$(IFS=' '; echo "${tests_subset[@]}")
  # execute the test in the background and redirect to a log file that buffers test output
  python ${defaults} "$tests_batch" 2>&1 > "parallel_test_output-$i.txt" &
  test_ids+=($i) # save the test's id in an array with running tests
  pids+=($!) # save the PID in an array with running tests
done

status=0 # reset the script status
printf "Waiting for batch to finish: $(IFS=' '; echo "${pids[@]}")\n"
# wait for running tests
for i in "${!pids[@]}"; do
  pid=${pids[$i]} # restore the particular PID
  printf "Waiting for batch $i >> parallel_test_output-$i.txt (PID: $pid)\n"
  wait -n $pid
  # get the exit status of the test
  test_status=$?
  # show the output of the failed test
  cat "parallel_test_output-$i.txt"
  if [[ $test_status != 0 ]]; then
    # Process exited with a non-zero exit status
    status=$test_status
  fi
done

# exit with the worse test result
exit $status
