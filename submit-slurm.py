#!/usr/bin/python

import os
import sys
import subprocess
import tempfile
import yaml


def makejob_train(commit_id, config_path, source_path, dataset_path):
    # Determine the rsync command based on the presence of "MSTAR" in source_path
    rsync_command = (
        f"rsync -r {source_path}/* {dataset_path}"
        if "MSTAR" or "2Shapes" or "3Shapes" or "MNIST_Shape" or "S1SLC" in source_path
        else f"rsync {source_path}/* {dataset_path}"
    )

    return f"""#!/bin/bash
 
#SBATCH --job-name=polsf
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --tmp=150G
#SBATCH --partition=gpu
#SBATCH --exclude=ruche-gpu06
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%j.out
#SBATCH --error=logslurms/slurm-%j.err
 
module load python/3.9.10/gcc-11.2.0
 
current_dir=`pwd`
export PATH=$PATH:~/.local/bin
 
echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
 
echo "Running on " $(hostname)
 
echo "Copying the source directory and data"
date
 
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs --exclude venv . $TMPDIR/code
 
mkdir {dataset_path}
{rsync_command}
 
 
echo "Checking out the correct version of the code commit_id {commit_id}"
cd $TMPDIR/code
git checkout {commit_id}
 
echo "Setting up the virtual environment"
 
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
 
# Install the library
python -m pip install .

echo "Training"

cd $current_dir
mv {config_path} $TMPDIR/code/configs.yml
python -m torchtmpl.main train $TMPDIR/code/configs.yml job_id/${{SLURM_JOB_ID}}

if [[ $? != 0 ]]; then
    exit -1
fi
"""


def makejob_test_retrain_without_logdir(commit_id, source_path, dataset_path, job_id, command):
    # Determine the rsync command based on the presence of "MSTAR" in source_path
    rsync_command = (
        f"rsync -r {source_path}/* {dataset_path}"
        if "MSTAR" or "2Shapes" or "3Shapes" or "MNIST_Shape" or "S1SLC" in source_path
        else f"rsync {source_path}/* {dataset_path}"
    )

    sbatch = f"""#!/bin/bash
 
#SBATCH --job-name={job_id}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --tmp=150G
#SBATCH --partition=gpu
#SBATCH --exclude=ruche-gpu06
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%j.out
#SBATCH --error=logslurms/slurm-%j.err
#SBATCH --dependency=afterany:{job_id}

module load python/3.9.10/gcc-11.2.0
 
current_dir=`pwd`
export PATH=$PATH:~/.local/bin
 
echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
 
echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
 
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs --exclude venv . $TMPDIR/code
 
mkdir {dataset_path}
{rsync_command}
 
 
echo "Checking out the correct version of the code commit_id {commit_id}"
cd $TMPDIR/code
git checkout {commit_id}
 
echo "Setting up the virtual environment"
 
python3 -m venv venv
source venv/bin/activate
 
# Install the library
python -m pip install .
 
echo "Training"

cd $current_dir

logdir=`cat job_id/{job_id}`
rm -f job_id/{job_id}
python -m torchtmpl.main {command} $logdir


if [[ $? != 0 ]]; then
    exit -1
fi
"""
    return sbatch

def makejob_test_retrain_with_logdir(commit_id, logdir, source_path, command, dataset_path):

    # Determine the rsync command based on the presence of "MSTAR" in source_path
    rsync_command = (
        f"rsync -r {source_path}/* {dataset_path}"
        if "MSTAR" or "2Shapes" or "3Shapes" or "MNIST_Shape" or "S1SLC" in source_path
        else f"rsync {source_path}/* {dataset_path}"
    )

    return f"""#!/bin/bash
 
#SBATCH --job-name=monjob
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --tmp=150G
#SBATCH --partition=gpu
#SBATCH --exclude=ruche-gpu06
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%j.out
#SBATCH --error=logslurms/slurm-%j.err
 
module load python/3.9.10/gcc-11.2.0
 
current_dir=`pwd`
export PATH=$PATH:~/.local/bin
 
echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
 
echo "Running on " $(hostname)
 
echo "Copying the source directory and data"
date
 
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs --exclude venv . $TMPDIR/code
 
mkdir {dataset_path}
{rsync_command}
 
 
echo "Checking out the correct version of the code commit_id {commit_id}"
cd $TMPDIR/code
git checkout {commit_id}
 
echo "Setting up the virtual environment"
 
python3 -m venv venv
source venv/bin/activate
 
# Install the library
python -m pip install .
 
echo "Training"

cd $current_dir

python -m torchtmpl.main {command} {logdir}
 

if [[ $? != 0 ]]; then
    exit -1
fi
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    job_id = subprocess.check_output("sbatch job.sbatch | cut -d \" \" -f 4", shell=True)
    print(f"Submitted job {job_id.decode().strip()}")
    return job_id.decode().strip()


# Ensure all the modified files have been staged and commited
# This is to guarantee that the commit id is a reliable certificate
# of the version of the code you want to evaluate
result = int(
    subprocess.run(
        "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.decode()
)
"""
if result > 0:
    print(f"We found {result} modifications either not staged or not commited")
    raise RuntimeError(
        "You must stage and commit every modification before submission "
    )
"""
commit_id = subprocess.check_output(
    "git log --pretty=format:'%H' -n 1", shell=True
).decode()

print(f"I will be using the commit id {commit_id}")

# Ensure the log directory exists
os.system("mkdir -p logslurms")

if len(sys.argv) not in [5,4]:
    print(f"Usage : ")
    print(f" {sys.argv[0]} train config.yaml sourcepath <numruns>")
    print(f" {sys.argv[0]} <retrain|test> logdir sourcepath")
    sys.exit(-1)


command = sys.argv[1]
if command == "train":
    config_path = sys.argv[2]
    source_path = sys.argv[3]
    if len(sys.argv) == 5:
        numruns = int(sys.argv[4])
    else:
        numruns = 1

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        dataset_path = config["data"]["dataset"]["trainpath"]

    # Copy the config in a temporary config file
    os.system("mkdir -p configs")
    tmp_config_path = tempfile.mkstemp(dir="./configs", suffix="-config.yml")[1]
    os.system(f"cp {config_path} {tmp_config_path}")

    os.system("mkdir -p job_id")

    # Launch the batch job
    job = makejob_train(commit_id, config_path=tmp_config_path, source_path=source_path, dataset_path=dataset_path)
    job_id = submit_job(job)
    for i in range(numruns - 1):
        job = makejob_test_retrain_without_logdir(commit_id, source_path=source_path, command="retrain", job_id=job_id, dataset_path=dataset_path)
        job_id = submit_job(job)
    job = makejob_test_retrain_without_logdir(commit_id, source_path=source_path, command="test", job_id=job_id, dataset_path=dataset_path)
    job_id = submit_job(job)
else:
    logdir = sys.argv[2]
    source_path = sys.argv[3]
    job = makejob_test_retrain_with_logdir(commit_id, logdir, source_path, command=command, job_id=None)
    job_id = submit_job(job)
