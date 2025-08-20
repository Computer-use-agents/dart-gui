class MockTaskLoader:
    def poll_for_tasks(self):
        # 这里实现实际的任务获取逻辑
        return [{"task_id": i} for i in range(10)] 

async def main():
    ray.init()
    coordinator = AgentCoordinator(max_concurrent_envs=32)
    task_loader = MockTaskLoader()
    await coordinator.start_rollout(task_loader)

asyncio.run(main())