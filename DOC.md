Here is a structured `README.md` based on your instructions. All Chinese comments and descriptions have been translated into English.

-----

# DART-GUI Training & Rollout System Setup

This guide provides instructions for setting up the DART-GUI environment, including Docker container initialization, database schema configuration, and execution scripts for sampling and training.

## 1\. Preparation

### Download Docker Images

Pull the required images for the GUI agent and the database.

```bash
# DART-GUI Image
docker pull crpi-iwtwdoj3ikoon38c.cn-beijing.personal.cr.aliyuncs.com/pengxiangli1999/dart-gui:v0

# MySQL Image
docker pull mysql:8.0.44-debian
```

### Prepare Model Checkpoints

Download the `UI-TARS-1.5-7B` model using the HuggingFace CLI. Replace `<your local path>` with your actual directory.

```bash
huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B --local-dir <your local path>
```

-----

## 2\. Docker Initialization

Initialize the containers on your **GPU Machine**. Ensure you replace specific paths (like MySQL volume) with your local paths.

### Rollouter Container

Used for the rollout service.

```bash
docker run -dit \
  --name rollouter \
  --gpus all \
  -p 6008:6008 \
  -p 8881:8881 \
  -p 15959:15959 \
  --shm-size=200g \
  crpi-iwtwdoj3ikoon38c.cn-beijing.personal.cr.aliyuncs.com/pengxiangli1999/dart-gui:v0
```

### Trainer Container

Used for model training.

```bash
docker run -dit \
  --name trainer \
  --gpus all \
  -p 6009:6008 \
  -p 8882:8881 \
  -p 15960:15959 \
  --shm-size=1500g \
  crpi-iwtwdoj3ikoon38c.cn-beijing.personal.cr.aliyuncs.com/pengxiangli1999/dart-gui:v0
```

### MySQL Container

Database server for tracking runs and checkpoints. Replace `<your sql default path>` with your local storage path.

```bash
docker run -dit \
  --name mysql-server \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=admin \
  -v <your sql default path>:/var/lib/mysql \
  mysql:8.0.44-debian
```

-----

## 3\. Database Configuration

Connect to your MySQL container and initialize the tables using the SQL below.

**Database Credentials:**

  * **User:** root
  * **Password:** admin
  * **Port:** 3306

### SQL Schema

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

-----

## 4\. Environment Setup (CPU Machine)

Please follow the instructions in the repository below to initialize the environment on your **CPU machine**:

  * **Repository:** [GUI-Docker-Env](https://github.com/Computer-use-agents/GUI-Docker-Env.git)

-----

## 5\. Execution

### Step 1: Start Rollouter (GPU Machine)

Inside the `rollouter` docker container:

```bash
sh model_service.sh
```

### Step 2: Start Agent Runner (CPU Machine)

Inside the configured CPU environment:

```bash
sh run.sh
```

### Step 3: Start Training (GPU Machine)

Once the links are set up and data is flowing, start the training process inside the `trainer` docker container:

```bash
sh examples/osworld/async/run_trainer_debug_w_rollout_stepwise_train_pt.sh
```

### Would you like me to create a `docker-compose.yml` file to automate the container creation process?