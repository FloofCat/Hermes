# When Less is More: Achieving Faster Convergence in Distributed Edge ML

This repo is an official implementation of **Hermes** (currently under review at the 31st IEEE HiPC). 

Achieving fast convergence in Distributed Machine Learning (DML) in resource-constrained edge devices is crucial for real-world applications. This paper proposes a novel probabilistic framework, Hermes, that accelerates DML by focusing on "major updates." Hermes leverages a dynamic threshold based on recent test loss behavior to identify statistically significant improvements in the model's generalization capability. This allows nodes to transmit updates only when a major improvement is detected, reducing communication overhead. Additionally, Hermes addresses the inefficiency of static dataset sizes in state-of-the-art protocols by dynamically allocating datasets to nodes based on their memory capacity. This approach improves resource utilization and avoids performance degradation due to straggler nodes, along with several optimizations in order to further convergence acceleration and communication reduction. We demonstrate through experiments that Hermes achieves faster convergence compared to existing methods, resulting in a training time reduction of 12.11x with similar or better accuracy.

## 0. Installation
- You will require a parameter server (PS) and X number of worker nodes. To connect your workers and PS to a private network (unless otherwise connected on LAN or a central network), we recommend using Tailscale (https://tailscale.com/) for doing so.
- Clone this repository to your PS.
- Modify the configuration files in `./Hermes/settings/`. This folder contains `config.json, master.json, settings.json`. You are required to add the IPs, hostnames and passwords of the worker and the PS in order to access the workers and the master itself.
- You may transfer files to workers via `scp` using `./transfer.sh`. If you are unable to execute this bash file, change the permissions of this file to allow execution -- `chmod +x ./transfer.sh`.

### 0a. PS Installation
- You may install all the required dependencies on the PS via the following command. Change line 38 to your PS' IP on the private network.
```bash
./dependencies_master.sh
```

### 0b. Worker Installation
- You may install all the required dependencies on each worker via the following command.
```bash
./dependencies_master.sh
```

## 1. Execution
The novelty of Hermes is we allow users to utilize their own models and datasets for training the same via DML. You may define your own custom model and dataset in `./examples/master.py`. You need not modify `./examples/worker.py`.

An example has been provided in the comments of `./examples/master.py`. Once modified, run the following commands on your respective devices.

### 1a. Execution on PS
Execution on the PS can be done using the following commands. You would need to initialize the Kafka broker and execute the script you modified.
```bash
tmux new-session -d -s kafkaone "cd ./kafka_2.13-3.7.0/ && bin/zookeeper-server-start.sh config/zookeeper.properties"
tmux new-session -d -s kafkatwo "cd ./kafka_2.13-3.7.0/ && bin/kafka-server-start.sh config/server.properties"
sleep 10
tmux new-session -d -s master "cd ./examples/ && python3 master.py"
```

You can review the logs of execution during training. You can do so by the following command.
```bash
tail -f ./Hermes/data/centralized-logs/distml-central.txt
```

### 1b. Execution on Workers
To begin execution on workers, you will need to wait until the logs on the PS state the following: 
> "Master has been successfully initialized!" 

Execution can be done using the following command.
```bash
tmux new-session -d -s worker "cd ./examples/ && python3 worker.py"
```

### 1c. Begin training on PS
Once the master and workers have been initialized, you will need to input 'y' to the master twice to send the model to the workers as well as the first dataset batch. 

## 2. License
Our source code is under the GNU General Public License v3.0.

## 3. Authors
Authors and other contributors will be added after acceptance.


