#!/bin/bash

set -e


GID=$(id -g "${USER}")

if [[ -z "${TT_METAL_HOME}" ]]; then
  TT_METAL_HOME=$(git rev-parse --show-toplevel)
fi

function run_docker_common {

    # Check if /dev/tenstorrent exists and add it to the docker run command if it does
    local tt_device=""
    if [[ -e /dev/tenstorrent ]]; then
        tt_device="--device /dev/tenstorrent"
    fi



    docker run \
        --rm \
        -v ${TT_METAL_HOME}:/${TT_METAL_HOME} \
        -v /home:/home \
        -v /dev/hugepages-1G:/dev/hugepages-1G \
        -v /etc/group:/etc/group:ro \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/shadow:/etc/shadow:ro \
        -v /mnt/MLPerf:/mnt/MLPerf \
        -w ${TT_METAL_HOME} \
        -e TT_METAL_HOME=${TT_METAL_HOME} \
        -e LOGURU_LEVEL=${LOGURU_LEVEL} \
        -e LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
        -e ARCH_NAME=${ARCH_NAME} \
        -e PYTHONPATH=${TT_METAL_HOME} \
        -e SILENT=${SILENT} \
		    -e VERBOSE=${VERBOSE} \
        -u ${UID}:${GID} \
        ${tt_device} \
        --net host \
        --privileged \
        "${docker_opts[@]}" \
        ${TT_METAL_DOCKER_IMAGE_TAG} \
        "${cmd[@]}"
}
