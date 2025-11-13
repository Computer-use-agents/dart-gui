# MySQL Datasets 管理模块使用指南

## 概述

这是一个基于SQLAlchemy ORM的MySQL数据集管理模块，专门用于管理强化学习训练过程中的轨迹数据。该模块提供了完整的CRUD操作、会话管理、事务支持等功能。

## 核心功能

- ✅ **ORM数据模型**: 基于SQLAlchemy的Dataset模型
- ✅ **会话管理**: 自动处理数据库连接和事务
- ✅ **CRUD操作**: 完整的增删改查功能
- ✅ **批量操作**: 支持批量查询和删除
- ✅ **统计查询**: 提供使用统计和聚合查询
- ✅ **条件搜索**: 支持多条件组合搜索
- ✅ **错误处理**: 完善的异常处理机制

## 数据库表结构

### datasets 表

| 字段名 | 类型 | 说明 | 默认值 | 约束 |
|--------|------|------|--------|------|
| `id` | `INTEGER` | 自增主键 | - | `PRIMARY KEY, AUTO_INCREMENT` |
| `trajectory_id` | `VARCHAR(255)` | 轨迹ID | - | `NOT NULL, UNIQUE` |
| `created_at` | `TIMESTAMP` | 创建时间 | `CURRENT_TIMESTAMP` | - |
| `used` | `INTEGER` | 使用次数 | `0` | - |
| `model_version` | `INTEGER` | 模型版本 | `0` | - |
| `run_id` | `VARCHAR(255)` | 运行ID | `NULL` | - |

### 索引
- `idx_run_id`: 在 `run_id` 字段上的索引，用于加速按运行ID查询

## 快速开始

### 1. 安装依赖

```bash
pip install sqlalchemy pymysql
```

### 2. 基本使用

```python
from verl.utils.database.mysql import create_database_manager

# 创建数据库管理器
manager = create_database_manager()

# 创建数据集记录
dataset = manager.create_dataset(
    trajectory_id="traj_001",
    run_id="experiment_alpha",
    used=0
)
print(f"创建成功: {dataset}")
```

## 详细API文档

### 初始化

```python
from verl.utils.database.mysql import MySQLDatasetsORM, DB_CONFIG

# 方式1: 使用预定义配置
manager = MySQLDatasetsORM(DB_CONFIG)

# 方式2: 使用自定义配置
custom_config = {
    'host': 'your_host',
    'user': 'your_user',
    'password': 'your_password',
    'database': 'your_database',
    'port': 3306,
    'charset': 'utf8mb4'
}
manager = MySQLDatasetsORM(custom_config)

# 方式3: 使用工厂函数
manager = create_database_manager()
```

### 创建操作

#### `create_dataset(trajectory_id, run_id, used=0)`

创建新的数据集记录。

**参数:**
- `trajectory_id` (str): 轨迹ID，必须唯一
- `run_id` (str): 运行ID
- `used` (int, 可选): 使用次数，默认0

**返回:**
- `Dict[str, Any]`: 创建的数据记录字典

**示例:**
```python
# 创建基础记录
dataset = manager.create_dataset("traj_001", "run_alpha")

# 创建带使用次数的记录
dataset = manager.create_dataset("traj_002", "run_alpha", used=5)

# 处理重复ID错误
try:
    dataset = manager.create_dataset("traj_001", "run_beta")
except ValueError as e:
    print(f"创建失败: {e}")  # trajectory_id 'traj_001' already exists
```

### 查询操作

#### `get_dataset_by_id(dataset_id)`

根据ID查询单个记录。

**参数:**
- `dataset_id` (int): 数据集ID

**返回:**
- `Optional[Dict[str, Any]]`: 数据记录字典，不存在时返回None

#### `get_dataset_by_trajectory_id(trajectory_id)`

根据轨迹ID查询单个记录。

**参数:**
- `trajectory_id` (str): 轨迹ID

**返回:**
- `Optional[Dict[str, Any]]`: 数据记录字典，不存在时返回None

#### `get_datasets_by_run_id(run_id, offset=0, limit=1)`

根据运行ID查询记录列表，支持分页。

**参数:**
- `run_id` (str): 运行ID
- `offset` (int, 可选): 偏移量，默认0
- `limit` (int, 可选): 限制数量，默认1

**返回:**
- `List[Dict[str, Any]]`: 数据记录字典列表

**示例:**
```python
# 获取运行alpha的所有记录
datasets = manager.get_datasets_by_run_id("run_alpha")

# 分页获取
datasets = manager.get_datasets_by_run_id("run_alpha", offset=10, limit=20)
```

#### `get_all_datasets(limit=100, offset=0)`

获取所有记录，支持分页。

**参数:**
- `limit` (int, 可选): 限制数量，默认100
- `offset` (int, 可选): 偏移量，默认0

**返回:**
- `List[Dict[str, Any]]`: 数据记录字典列表

### 更新操作

#### `update_used(trajectory_id, used)`

更新使用次数。

**参数:**
- `trajectory_id` (str): 轨迹ID
- `used` (int): 新的使用次数

**返回:**
- `bool`: 是否更新成功

#### `update_run_id(trajectory_id, new_run_id)`

更新运行ID。

**参数:**
- `trajectory_id` (str): 轨迹ID
- `new_run_id` (str): 新的运行ID

**返回:**
- `bool`: 是否更新成功

#### `update_dataset(trajectory_id, **kwargs)`

通用更新方法，可更新多个字段。

**参数:**
- `trajectory_id` (str): 轨迹ID
- `**kwargs`: 要更新的字段和值

**返回:**
- `bool`: 是否更新成功

**示例:**
```python
# 更新单个字段
success = manager.update_used("traj_001", 10)

# 更新多个字段
success = manager.update_dataset(
    trajectory_id="traj_001",
    used=15,
    model_version=2,
    run_id="run_beta"
)
```

### 删除操作

#### `delete_dataset(trajectory_id)`

删除单个记录。

**参数:**
- `trajectory_id` (str): 轨迹ID

**返回:**
- `bool`: 是否删除成功

#### `delete_datasets_by_run_id(run_id)`

根据运行ID批量删除记录。

**参数:**
- `run_id` (str): 运行ID

**返回:**
- `int`: 删除的记录数量

### 统计和搜索

#### `count_datasets()`

统计总记录数。

**返回:**
- `int`: 记录总数

#### `get_usage_stats()`

获取使用统计信息。

**返回:**
- `Dict[str, Any]`: 包含统计信息的字典

**返回字段:**
- `total_count`: 总记录数
- `avg_usage_time`: 平均使用次数
- `max_usage_time`: 最大使用次数
- `min_usage_time`: 最小使用次数
- `unique_runs`: 唯一运行ID数量

#### `search_datasets(**filters)`

条件搜索记录。

**支持的过滤条件:**
- `run_id`: 运行ID
- `usage_time_min`: 最小使用次数
- `usage_time_max`: 最大使用次数
- `created_after`: 创建时间下限
- `created_before`: 创建时间上限

**返回:**
- `List[Dict[str, Any]]`: 符合条件的数据记录列表

**示例:**
```python
# 搜索高使用次数的记录
high_usage = manager.search_datasets(usage_time_min=10)

# 搜索特定运行ID的记录
run_records = manager.search_datasets(run_id="experiment_alpha")

# 搜索时间范围内的记录
from datetime import datetime
recent_records = manager.search_datasets(
    created_after=datetime(2024, 1, 1),
    created_before=datetime(2024, 12, 31)
)

# 组合搜索条件
filtered = manager.search_datasets(
    run_id="experiment_alpha",
    usage_time_min=5,
    usage_time_max=20
)
```

## 实际应用场景

### 1. 强化学习训练数据管理

```python
def manage_training_data():
    manager = create_database_manager()
    
    # 创建训练轨迹
    trajectory_id = f"traj_{int(time.time())}"
    dataset = manager.create_dataset(
        trajectory_id=trajectory_id,
        run_id="rl_training_v1",
        used=0
    )
    
    # 训练完成后更新使用次数
    manager.update_used(trajectory_id, 1)
    
    # 查询可用于训练的数据
    available_data = manager.search_datasets(
        run_id="rl_training_v1",
        usage_time_min=0,
        usage_time_max=5  # 使用次数不超过5次
    )
    
    return available_data
```

### 2. 实验数据追踪

```python
def track_experiment_data():
    manager = create_database_manager()
    
    # 记录实验开始
    experiment_id = "exp_2024_001"
    
    # 创建多个数据点
    for i in range(10):
        trajectory_id = f"{experiment_id}_traj_{i}"
        manager.create_dataset(
            trajectory_id=trajectory_id,
            run_id=experiment_id,
            used=0
        )
    
    # 获取实验统计信息
    experiment_data = manager.get_datasets_by_run_id(experiment_id)
    stats = manager.get_usage_stats()
    
    print(f"实验 {experiment_id} 包含 {len(experiment_data)} 个数据点")
    print(f"总体统计: {stats}")
```

### 3. 数据清理和维护

```python
def cleanup_old_data():
    manager = create_database_manager()
    
    # 删除过期的实验数据
    old_experiments = ["exp_2023_001", "exp_2023_002"]
    for exp_id in old_experiments:
        deleted_count = manager.delete_datasets_by_run_id(exp_id)
        print(f"删除了实验 {exp_id} 的 {deleted_count} 条记录")
    
    # 清理使用次数过多的数据
    overused_data = manager.search_datasets(usage_time_min=100)
    for data in overused_data:
        manager.delete_dataset(data['trajectory_id'])
        print(f"删除了过度使用的数据: {data['trajectory_id']}")
```

## 错误处理最佳实践

### 1. 创建操作错误处理

```python
def safe_create_dataset(trajectory_id, run_id, used=0):
    manager = create_database_manager()
    
    try:
        dataset = manager.create_dataset(trajectory_id, run_id, used)
        return dataset
    except ValueError as e:
        # 处理重复ID错误
        print(f"创建失败 - 重复ID: {e}")
        return None
    except Exception as e:
        # 处理其他数据库错误
        print(f"创建失败 - 数据库错误: {e}")
        return None
```

### 2. 查询操作错误处理

```python
def safe_get_dataset(trajectory_id):
    manager = create_database_manager()
    
    try:
        dataset = manager.get_dataset_by_trajectory_id(trajectory_id)
        if dataset is None:
            print(f"数据不存在: {trajectory_id}")
            return None
        return dataset
    except Exception as e:
        print(f"查询失败: {e}")
        return None
```

### 3. 批量操作错误处理

```python
def safe_batch_update(trajectory_ids, new_used):
    manager = create_database_manager()
    
    success_count = 0
    for trajectory_id in trajectory_ids:
        try:
            success = manager.update_used(trajectory_id, new_used)
            if success:
                success_count += 1
        except Exception as e:
            print(f"更新 {trajectory_id} 失败: {e}")
    
    print(f"成功更新 {success_count}/{len(trajectory_ids)} 条记录")
    return success_count
```

## 性能优化建议

### 1. 批量操作

```python
# 避免循环单个操作
# ❌ 不推荐
for trajectory_id in trajectory_ids:
    manager.update_used(trajectory_id, 1)

# ✅ 推荐：使用事务批量操作
def batch_update_usage(trajectory_ids, new_used):
    manager = create_database_manager()
    with manager.get_session() as session:
        for trajectory_id in trajectory_ids:
            dataset = session.query(Dataset).filter(
                Dataset.trajectory_id == trajectory_id
            ).first()
            if dataset:
                dataset.used = new_used
```

### 2. 合理使用分页

```python
# 处理大量数据时使用分页
def process_all_datasets():
    manager = create_database_manager()
    offset = 0
    limit = 100
    
    while True:
        datasets = manager.get_all_datasets(limit=limit, offset=offset)
        if not datasets:
            break
        
        # 处理这批数据
        for dataset in datasets:
            process_dataset(dataset)
        
        offset += limit
```

### 3. 索引优化

确保在常用查询字段上创建索引：

```sql
-- 在run_id字段上创建索引（已存在）
CREATE INDEX idx_run_id ON datasets(run_id);

-- 如果需要按使用次数查询，可以添加索引
CREATE INDEX idx_used ON datasets(used);

-- 如果需要按创建时间查询，可以添加索引
CREATE INDEX idx_created_at ON datasets(created_at);
```

## 监控和日志

### 1. 启用详细日志

```python
import logging

# 设置详细日志
logging.getLogger('verl.utils.database.mysql').setLevel(logging.DEBUG)

# 或者只记录SQL语句
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### 2. 性能监控

```python
import time

def monitor_database_performance():
    manager = create_database_manager()
    
    # 监控查询性能
    start_time = time.time()
    datasets = manager.get_all_datasets(limit=1000)
    query_time = time.time() - start_time
    
    print(f"查询1000条记录耗时: {query_time:.3f}秒")
    
    # 监控统计查询性能
    start_time = time.time()
    stats = manager.get_usage_stats()
    stats_time = time.time() - start_time
    
    print(f"统计查询耗时: {stats_time:.3f}秒")
```

## 常见问题解答

### Q1: 如何处理数据库连接超时？

**A:** 模块已配置连接池和预检查功能，如果仍遇到超时问题：

```python
# 自定义连接配置
custom_config = {
    **DB_CONFIG,
    'connect_timeout': 60,
    'read_timeout': 60,
    'write_timeout': 60
}
manager = MySQLDatasetsORM(custom_config)
```

### Q2: 如何备份和恢复数据？

**A:** 使用MySQL的备份工具：

```bash
# 备份
mysqldump -h 112.125.88.107 -u agentictrl -p BIGAI datasets > datasets_backup.sql

# 恢复
mysql -h 112.125.88.107 -u agentictrl -p BIGAI < datasets_backup.sql
```

### Q3: 如何处理大量数据的导入？

**A:** 使用批量插入：

```python
def bulk_insert_datasets(datasets_list):
    manager = create_database_manager()
    with manager.get_session() as session:
        for dataset_data in datasets_list:
            dataset = Dataset(**dataset_data)
            session.add(dataset)
        session.commit()
```

### Q4: 如何实现数据迁移？

**A:** 使用SQLAlchemy的迁移工具或自定义迁移脚本：

```python
def migrate_data():
    manager = create_database_manager()
    
    # 获取所有需要迁移的数据
    datasets = manager.get_all_datasets()
    
    for dataset in datasets:
        # 执行迁移逻辑
        if dataset['model_version'] == 0:
            manager.update_dataset(
                dataset['trajectory_id'],
                model_version=1
            )
```

## 总结

MySQL Datasets管理模块提供了完整的数据库操作功能，适用于强化学习训练过程中的数据管理。通过合理使用API和遵循最佳实践，可以高效地管理大量训练数据，支持复杂的查询和统计需求。

关键要点：
1. 使用上下文管理器自动处理会话
2. 合理使用分页处理大量数据
3. 实现适当的错误处理机制
4. 根据查询模式优化索引
5. 定期监控性能和日志 