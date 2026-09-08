#!/bin/bash
#SBATCH --job-name=build-op-scvi-sif
#SBATCH --output=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/openproblems_scvi_validation/build-%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/openproblems_scvi_validation/build-%j.err
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread

set -eo pipefail
source /etc/profile

root=/lustre/fsn1/projects/rech/xeg/uat95fg/containers
singularity_bin=/gpfslocalsys/singularity/singularity-3.8.5/bin/singularity
mkdir -p /lustre/fsn1/projects/rech/xeg/uat95fg/scprint_data/openproblems_scvi_validation

# `singularity build <sif> <sandbox>` first copies every file from Lustre to a
# temporary rootfs before compression.  This sandbox contains many small Python
# files, making that metadata-heavy copy slower than the compile partition time
# limit.  Squash the sandbox directly, then wrap that system partition in SIF.
job_tmp="${JOBSCRATCH:-/lustre/fsn1/jobscratch/${USER}_${SLURM_JOB_ID}}"
squashfs="${job_tmp}/openproblems_base_pytorch_nvidia_1.squashfs"
output_tmp="${job_tmp}/openproblems_base_pytorch_nvidia_1.sif"
output="$root/openproblems_base_pytorch_nvidia_1.sif"
mkdir -p "$job_tmp"

mksquashfs \
  "$root/openproblems_base_pytorch_nvidia_1.sandbox" \
  "$squashfs" \
  -noappend -comp gzip -processors 8 -mem 16G -no-xattrs

rm -f "$output_tmp"
"$singularity_bin" sif new "$output_tmp"
"$singularity_bin" sif add \
  --datatype 4 --parttype 2 --partfs 1 --partarch 2 \
  "$output_tmp" "$squashfs"
mv -f "$output_tmp" "$output"
"$singularity_bin" sif list "$output"
