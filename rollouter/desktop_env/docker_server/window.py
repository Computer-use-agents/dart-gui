"""
可视化窗口路由
- 提供 /window 页面，用于从后端 /emulators 读取可用 emulator，并在页面中通过 iframe 嵌入对应的 VNC 实时画面。
- UI 与数据获取逻辑在 templates/window.html 中实现，这里仅注册 Flask 路由。
"""

from typing import Any
from flask import Flask, render_template


def register_window_routes(app: Flask) -> None:
    """
    注册可视化窗口路由。

    GET /window
      - 返回 window.html 模板
      - 模板负责：调用 /emulators 获取列表，选择 emulator 后将 iframe.src 指向 http(s)://{当前主机}:{vnc_port}
    """
    @app.route("/window", methods=["GET"])
    def emulator_window() -> Any:
        return render_template("window.html")


__all__ = ["register_window_routes"]
