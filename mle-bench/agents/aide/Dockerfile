FROM johnsonzheng03/predict-before-execute

# where to put submission.csv, will be extracted
ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}
# name of the conda environment to use
ARG CONDA_ENV_NAME=agent

# put all the agent files in the expected location
COPY . ${AGENT_DIR}