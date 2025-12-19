# <img src="https://computer-use-agents.github.io/dart-gui/icon.png" alt="DART-GUI" style="height: 30px;"> DART-GUI
### Efficient Multi-turn RL for GUI Agents via Decoupled Training and Adaptive Data Curation

<p align="center">
&nbsp&nbsp🌐 <a href="https://computer-use-agents.github.io/dart-gui/">Website</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2509.23866">arXiv</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://huggingface.co/dart-gui/dart-gui-7b">Model</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/dart-gui/dart-gui-7b">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp📊 <a href="">Dataset (coming soon)</a>&nbsp&nbsp
</p>

<p align="center">
  <img src="https://computer-use-agents.github.io/dart-gui/stats/teaser.png" alt="DART-GUI" style="height: 400px;">
</p>

## 📢 Updates
- [2025-12-10] We released training code, sampling code, SQL schema, and related Docker configurations.
- [2025-11-30] We released ENV code and Docker setup.
- [2025-10-30] We released checkpoint and inference code.
- [2025-09-30] We released our [paper](https://arxiv.org/abs/2509.23866) and [project page](https://computer-use-agents.github.io/dart-gui). Check it out!

## 🔨 TODO
- [ ] Polish the codebase.
- [ ] Merge with the latest verl version.
- [x] Release the model checkpoint for DART-GUI-7B.
- [x] Add the training code and pipeline.
- [x] Release checkpoint and inference code.
- [x] Release ENV code and Docker setup.
- [x] Release training code and sampling code.
- [x] Release SQL schema and related Docker configurations.

## 🚀 Quick Start

This guide provides instructions for setting up the DART-GUI environment, including Docker container initialization, database schema configuration, and execution scripts for sampling and training.

### 1. Preparation

#### Download Docker Images

Pull the required images for the GUI agent and the database.

```bash
# DART-GUI Image
docker pull crpi-iwtwdoj3ikoon38c.cn-beijing.personal.cr.aliyuncs.com/pengxiangli1999/dart-gui:v0

# MySQL Image
docker pull mysql:8.0.44-debian
```

#### Prepare Model Checkpoints

Download the `UI-TARS-1.5-7B` model using the HuggingFace CLI. Replace `<your local path>` with your actual directory.

```bash
huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B --local-dir <your local path>
```

### 2. Docker Initialization

Initialize the containers on your **GPU Machine**. Ensure you replace specific paths (like MySQL volume) with your local paths.

#### Rollouter Container

Used for the rollout service.

```bash
docker run -dit \
  --name rollouter \
  --gpus all \
  -p 6008:6008 \
  -p 8881:8881 \
  -p 15959:15959 \
  --shm-size=200g \
  -v <your workspace>:<your workspace in docker> \
  crpi-iwtwdoj3ikoon38c.cn-beijing.personal.cr.aliyuncs.com/pengxiangli1999/dart-gui:v0
```

#### Trainer Container

Used for model training.

```bash
docker run -dit \
  --name trainer \
  --gpus all \
  -p 6009:6008 \
  -p 8882:8881 \
  -p 15960:15959 \
  -v <your workspace>:<your workspace in docker> \
  --shm-size=1500g \
  crpi-iwtwdoj3ikoon38c.cn-beijing.personal.cr.aliyuncs.com/pengxiangli1999/dart-gui:v0
```

#### MySQL Container

Database server for tracking runs and checkpoints. Replace `<your sql default path>` with your local storage path.

```bash
docker run -dit \
  --name mysql-server \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=admin \
  -v <your sql default path>:/var/lib/mysql \
  mysql:8.0.44-debian
```

### 3. Database Configuration

Connect to your MySQL container and initialize the tables using the SQL below.

**Database Credentials:**
- **User:** root
- **Password:** admin
- **Port:** 3306

#### SQL Schema

```sql
--
-- Table structure for table `rollout_run`
--

DROP TABLE IF EXISTS `rollout_run`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `rollout_run` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `run_id` varchar(191) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `trajectory_id` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `task_id` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `trace_id` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `split_dir` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `reward` double DEFAULT NULL,
  `num_chunks` int DEFAULT NULL,
  `used` int NOT NULL DEFAULT '0',
  `model_version` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `instruction` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `create_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_rollout_run_id` (`id`),
  UNIQUE KEY `uk_rollout_run_traj_run` (`trajectory_id`,`run_id`),
  KEY `idx_rollout_run_task` (`task_id`)
) ENGINE=InnoDB AUTO_INCREMENT=1319846 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `checkpoint`
--

DROP TABLE IF EXISTS `checkpoint`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `checkpoint` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT 'Primary Key ID',
  `name` varchar(50) NOT NULL DEFAULT '' COMMENT 'Checkpoint Name (Unique English Identifier)',
  `version` varchar(50) NOT NULL COMMENT 'Version Number (Semantic Versioning, e.g., v1.0.0)',
  `run_id` varchar(191) NOT NULL DEFAULT '',
  `status` varchar(20) NOT NULL DEFAULT 'PENDING' COMMENT 'Status: PENDING|RUNNING|COMPLETED|FAILED|DEPRECATED',
  `path` varchar(255) NOT NULL COMMENT 'Storage Path (e.g., s3://bucket/path/checkpoint.ckpt)',
  `source` varchar(50) DEFAULT NULL COMMENT 'Source (e.g., User Upload/Training Generated/System Migration)',
  `operator` varchar(50) DEFAULT NULL COMMENT 'Operator (User ID or System Account)',
  `remark` varchar(1024) DEFAULT NULL COMMENT 'Remark (Free text format)',
  `config_yaml` text COMMENT 'Full Deployment Config (Encrypted Storage)',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created At',
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last Updated At',
  `deleted_at` timestamp NULL DEFAULT NULL COMMENT 'Soft Delete Time',
  `started_at` timestamp NULL DEFAULT NULL COMMENT 'Started At',
  `finished_at` timestamp NULL DEFAULT NULL COMMENT 'Finished At',
  PRIMARY KEY (`id`),
  KEY `idx_status` (`status`),
  KEY `idx_created_at` (`created_at`),
  KEY `idx_updated_at` (`updated_at`)
) ENGINE=InnoDB AUTO_INCREMENT=3117 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='Model Checkpoint Table (Records training checkpoints and deployment versions)';
/*!40101 SET character_set_client = @saved_cs_client */;
```

### 4. Environment Setup (CPU Machine)

Please follow the instructions in the repository below to initialize the environment on your **CPU machine**:

- **Repository:** [GUI-Docker-Env](https://github.com/Computer-use-agents/GUI-Docker-Env.git)

### 5. Execution

#### Step 1: Start Rollouter (GPU Machine)

Inside the `rollouter` docker container:

```bash
cd dart_rollouter
sh model_service.sh
```

#### Step 2: Start Agent Runner (CPU Machine)

Inside the configured CPU environment:

```bash
cd dart_rollouter
sh run.sh
```

#### Step 3: Start Training (GPU Machine)

Once the links are set up and data is flowing, start the training process inside the `trainer` docker container:

```bash
sh examples/osworld/async/run_trainer_debug_w_rollout_stepwise_train_pt.sh
```

## 🤗 Team
### Core Contributors

<table>
<tr>
    <td align="center">
        <a href="https://github.com/Pengxiang-Li">
            <img src="https://github.com/Pengxiang-Li.png" width="75px;" alt="Pengxiang Li"/>
            <br />
            <sub><b>Pengxiang Li</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/huzechen123">
            <img src="https://github.com/huzechen123.png" width="75px;" alt="Zechen Hu"/>
            <br />
            <sub><b>Zechen Hu</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/shzirui">
            <img src="https://github.com/shzirui.png" width="75px;" alt="Zirui Shang"/>
            <br />
            <sub><b>Zirui Shang</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/RubyNg">
            <img src="https://github.com/RubyNg.png" width="75px;" alt="Jingrong Wu"/>
            <br />
            <sub><b>Jingrong Wu</b></sub>
        </a>
    </td>
</tr>
</table>

### Advisors
<table>
<tr>
    <td align="center">
        <a href="https://liqing.io">
            <img src="https://github.com/liqing-ustc.png" width="75px;" alt="Pengxiang Li"/>
            <br />
            <sub><b>Qing Li</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://zhigao2017.github.io/">
            <img src="https://github.com/zhigao2017.png" width="75px;" alt="Zhi Gao"/>
            <br />
            <sub><b>Zhi Gao</b></sub>
        </a>
    </td>
</tr>
</table>

**We are looking for collaborations and GPU resources support! If you are interested in agentic RL and GUI agents, feel free to [email](mailto:dylan.liqing@gmail.com) Qing Li!**

## 🤝 Acknowledgments
We thank the following open-source projects for making this work possible:
- [verl](https://github.com/volcengine/verl) for the excellent RL framework.
- [vLLM](https://github.com/vllm-project/vllm) for the fast inference engine.

We thank [DataCanvas Alaya](https://www.alayanew.com/) for the cloud computing and GPU support!

## 📝 Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝 :)

```bibtex
@article{li2025dart,
  title = {Efficient Multi-turn RL for GUI Agents via Decoupled Training and Adaptive Data Curation},
  author={Li, Pengxiang and Hu, Zechen and Shang, Zirui and Wu, Jingrong and Liu, Yang and Liu, Hui and Gao, Zhi and Shi, Chenrui and Zhang, Bofei and Zhang, Zihao and Shi, Xiaochuan and Yu, Zedong and Wu, Yuwei and Wu, Xinxiao and Jia, Yunde and Xiang, Liuyu and He, Zhaofeng and Li, Qing},
  journal={arXiv preprint arXiv:2509.23866},
  year={2025}
  url = {https://arxiv.org/abs/2509.23866}
}
```

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=computer-use-agents/dart-gui&type=Date)](https://star-history.com/#computer-use-agents/dart-gui&Date)
