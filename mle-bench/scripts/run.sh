export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent

IMAGE_ID=$(docker build -q --no-cache --platform=linux/amd64 -t ForeAgent/DeepSeek-V3.2 agents/aide --build-arg SUBMISSION_DIR=$SUBMISSION_DIR --build-arg LOGS_DIR=$LOGS_DIR --build-arg CODE_DIR=$CODE_DIR --build-arg AGENT_DIR=$AGENT_DIR)
echo $IMAGE_ID
python run_agent.py --agent-id ForeAgent/DeepSeek-V3.2 --competition-set experiments/splits/foreagent.txt --data-dir path/to/data/tasks/ data --gpu-device 0

