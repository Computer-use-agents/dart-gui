# SmartDataManager Tests

这个目录包含了 SmartDataManager 的单元测试和集成测试。

## 文件说明

- `test_smart_data_manager.py` - 完整的 pytest 单元测试套件
- `run_smart_manager_tests.py` - 简化的测试运行器，可以直接执行
- `README.md` - 本文件

## 运行测试

### 方式1：使用简化测试运行器（推荐）

这种方式不需要安装 pytest，可以直接运行：

```bash
cd tests
python run_smart_manager_tests.py
```

### 方式2：使用 pytest（完整测试）

如果你安装了 pytest，可以运行完整的测试套件：

```bash
# 安装 pytest（如果没有的话）
pip install pytest

# 运行所有测试
cd tests
pytest test_smart_data_manager.py -v

# 运行特定测试
pytest test_smart_data_manager.py::TestSmartDataManager::test_data_selection_priority -v
```

## 测试内容

### 核心功能测试

1. **基本功能测试**
   - SmartDataManager 初始化
   - 任务数量获取
   - 数据选择逻辑

2. **优先级排序测试**
   - model_version 优先级（降序）
   - used 计数优先级（升序）
   - reward 优先级（降序）

3. **数据获取测试**
   - 按索引获取数据（模拟 `Dataset.__getitem__`）
   - 无效索引处理
   - 空数据处理

4. **使用计数更新测试**
   - 实时更新轨迹的 used 计数
   - 处理不存在的轨迹
   - 数据库连接管理

5. **边界条件测试**
   - None 奖励值处理
   - 数据库连接错误处理
   - 空数据集处理

6. **统计功能测试**
   - 任务统计信息计算
   - 平均值和最大值计算

### 测试数据说明

测试使用模拟数据，包含以下轨迹：

```python
# 测试数据示例
trajectory_data = [
    {'trajectory_id': 'traj_001', 'model_version': 1, 'used': 2, 'reward': 0.8},
    {'trajectory_id': 'traj_002', 'model_version': 2, 'used': 1, 'reward': 0.7},
    {'trajectory_id': 'traj_003', 'model_version': 2, 'used': 0, 'reward': 0.9},  # 应该被优先选择
    {'trajectory_id': 'traj_004', 'model_version': 2, 'used': 0, 'reward': 0.6},  # 第二优先
]
```

### 预期选择顺序

根据优先级规则 `(model_version↓, used↑, reward↓)`：

1. `traj_003` - model_version=2, used=0, reward=0.9 （最优）
2. `traj_004` - model_version=2, used=0, reward=0.6 （次优）
3. `traj_002` - model_version=2, used=1, reward=0.7 （第三）
4. `traj_001` - model_version=1, used=2, reward=0.8 （最后）

## 测试输出示例

```
SmartDataManager Test Suite
==================================================

=== Testing Basic Functionality ===
✓ Created SmartDataManager with run_id='test_run', rollout_n=2
✓ Available tasks: 3
✓ Selected 2 trajectories for task_001
  First selected: traj_003 (mv=2, used=0, reward=0.9)
  Second selected: traj_004 (mv=2, used=0, reward=0.6)
✓ Priority selection working correctly
✓ Manager closed successfully
✅ test_basic_functionality PASSED

=== Testing Data by Index ===
✓ Index 0 returned 2 trajectories
✓ Invalid index 999 returned 0 trajectories
✓ Usage update calls verified
✅ test_data_by_index PASSED

=== Testing Priority Edge Cases ===
✓ Handled 3 trajectories with None rewards
✓ None reward handling works correctly
✅ test_priority_edge_cases PASSED

=== Testing Statistics ===
✓ Got statistics: {'total_trajectories': 4, 'avg_used': 0.75, 'avg_reward': 0.75, 'max_model_version': 2}
✓ Statistics calculation correct
✅ test_statistics PASSED

==================================================
Test Results: 4 passed, 0 failed
🎉 All tests passed!
```

## 添加新测试

如果你想添加新的测试用例，可以在 `test_smart_data_manager.py` 中添加新的测试方法：

```python
@patch('verl.utils.dataset.smart_data_manager.create_database_manager')
def test_your_new_feature(self, mock_create_db, mock_db_manager):
    """Test description"""
    # Your test code here
    pass
```

或者在 `run_smart_manager_tests.py` 中添加简单的测试函数。

## 故障排除

### 导入错误

如果遇到导入错误，确保：
1. 你在正确的目录下运行测试
2. `verl` 包在 Python 路径中
3. 所有依赖已安装

### 数据库连接错误

测试使用模拟数据库，不应该有真实的数据库连接。如果遇到数据库相关错误：
1. 检查 mock 是否正确配置
2. 确保没有真实的数据库调用

## 性能测试

对于性能测试，可以修改测试数据的规模：

```python
# 创建大量测试数据
large_dataset = [create_trajectory(i) for i in range(10000)]
```

这样可以测试在大数据集下的性能表现。