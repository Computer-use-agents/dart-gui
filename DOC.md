# Documentation For VeRL-OSWorld


### 和官方仓库进行同步
一个很好的实践是每隔一段时间尝试同步官方verl仓库的改动进入本仓库
本文档描述了如何将上游仓库（volcengine/verl）的最新更新同步到你的fork仓库（Computer-use-agents/verl）。
```bash
# 查看当前远程仓库配置
git remote -v

# 输出示例：
# origin  https://github.com/volcengine/verl.git (fetch)
# origin  https://github.com/volcengine/verl.git (push)
# origin_cua      git@github.com:Computer-use-agents/verl.git (fetch)
# origin_cua      git@github.com:Computer-use-agents/verl.git (push)
```

- `origin`: 上游仓库（volcengine/verl）
- `origin_cua`: 你的fork仓库（Computer-use-agents/verl）
如果你找不到两个remote，可以google下如何增加remote。
### 同步步骤

#### 1. 获取上游更新

```bash
# 从上游仓库获取最新代码
git fetch origin
```

#### 2. 切换到主分支

```bash
# 切换到主分支（通常是 main 或 master）
git checkout main
```

#### 3. 合并上游更新

选择以下任一方式：

方式一：使用 merge（推荐，保留完整历史）

```bash
# 将上游更新合并到当前分支
git merge origin/main
```

方式二：使用 rebase（保持提交历史整洁）

```bash
# 将当前分支的提交重新应用到上游更新之上
git rebase origin/main
```
注意，这一步一般会有冲突，一定要手动解除，并且尝试通过测试验证同步是否有效。
#### 4. 推送到你的fork

```bash
# 将更新推送到你的fork仓库
git push origin_cua main
```