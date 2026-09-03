"""控制台入口：读一行 → 跑完整任务链路 → 打印结果。

一个进程 = 一个会话，所以同一次运行里的多轮提问共享 WorkingMemory，"继续统计刚才的
结果"这类追问才能找到关联任务。

运行::

    python -m agent.cli            # 未配置 LLM_API_KEY 时自动用 MockLLM
    python -m agent.cli --verbose  # 额外打印状态流转、步骤与工具调用
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable

from agent.config import AppSettings, get_settings
from agent.enums import TaskStatus
from agent.errors import AgentError
from agent.models import Task, new_session_id
from agent.task_manager import TaskManager, build_task_manager

__all__ = ["ConsoleAgent", "main"]

_BANNER = """\
================================================================
 日志查询 Agent · 控制台
 直接输入你的问题，例如：
   查一下 order-service 的 ERROR 日志
   按服务统计刚才那批错误
 命令： /help 帮助   /history 本会话任务   /verbose 开关详情   /exit 退出
================================================================"""

_HELP = """\
可用命令：
  /help      显示这段帮助
  /history   列出本会话已完成的任务及其结论
  /verbose   开关"显示执行细节"
  /exit      退出（Ctrl-D 同效）"""


class ConsoleAgent:
    """把 TaskManager 包装成一问一答的控制台会话。"""

    def __init__(
        self,
        manager: TaskManager,
        *,
        session_id: str | None = None,
        verbose: bool = False,
        printer: Callable[[str], None] = print,
    ) -> None:
        self.manager = manager
        self.session_id = session_id or new_session_id()
        self.verbose = verbose
        self._emit = printer
        self._pending_clarification: str | None = None

    def emit(self, text: str) -> None:
        """输出一行。可注入 printer，测试里就能把整段会话收进列表断言。"""
        self._emit(text)

    # ---------------------------------------------------------------- 单轮

    def ask(self, text: str) -> Task:
        """跑一轮问答。上一轮若停在澄清态，这一轮的输入就是补充说明。"""
        if self._pending_clarification is not None:
            task_id = self._pending_clarification
            self._pending_clarification = None
            task = self.manager.provide_clarification(task_id, text)
        else:
            task = self.manager.create_task(text, session_id=self.session_id)
            task = self.manager.run(task.task_id)

        if task.status is TaskStatus.CLARIFYING:
            self._pending_clarification = task.task_id
        return task

    def render(self, task: Task) -> str:
        """把任务结果渲染成给用户看的文本。"""
        lines: list[str] = []
        if self.verbose:
            lines.extend(self._details(task))

        summary = task.summary
        if task.status is TaskStatus.CLARIFYING:
            lines.append(f"❓ {summary.output}")
        elif task.status is TaskStatus.COMPLETED:
            lines.append(f"✅ {summary.output}")
        else:
            lines.append(f"⚠️  任务未完成（{task.status.value}）：{summary.output}")
        return "\n".join(lines)

    def _details(self, task: Task) -> list[str]:
        window = self.manager.window(task.task_id)
        summary = task.summary
        lines = [
            f"— 任务 {task.task_id}",
            f"  意图：{summary.intent.value if summary.intent else '(未识别)'}"
            f"   状态：{summary.status.value}"
            f"   关联任务：{summary.related_task_ids or '无'}",
        ]
        for operation in summary.operations:
            lines.append(
                f"  步骤{operation.index + 1} [{operation.status.value}] {operation.description}"
            )
        for call_id, record in window.tool_results.items():
            flag = "ok" if record.ok else "失败"
            lines.append(f"  工具 {record.tool_name}({call_id}) → {flag}")
        return lines

    # ---------------------------------------------------------------- 交互

    def handle_command(self, text: str) -> bool:
        """处理 / 开头的命令；返回 True 表示要退出。"""
        command = text.strip().lower()
        if command in ("/exit", "/quit"):
            return True
        if command == "/help":
            self.emit(_HELP)
        elif command == "/verbose":
            self.verbose = not self.verbose
            self.emit(f"详情模式：{'开' if self.verbose else '关'}")
        elif command == "/history":
            self.emit(self._history())
        else:
            self.emit(f"未知命令：{text}（/help 查看可用命令）")
        return False

    def _history(self) -> str:
        working = self.manager.memory.working(self.session_id)
        task_ids = working.task_ids()
        if not task_ids:
            return "本会话还没有已完成的任务。"
        lines = []
        for index, task_id in enumerate(task_ids, start=1):
            summary = working.task_summary_history[task_id]
            lines.append(
                f"{index}. [{summary['status']}] {summary['content']}\n"
                f"   → {summary.get('output') or '(无输出)'}"
            )
        return "\n".join(lines)

    def run_forever(self, lines: Iterable[str] | None = None) -> None:
        """REPL 主循环。

        ``lines`` 是测试注入点：给定可迭代对象时逐行消费，不给则从 stdin 读。
        """
        reader = _line_reader(lines)
        self.emit(_BANNER)
        self.emit(self._llm_notice())

        while True:
            self.emit("")
            try:
                text = reader()
            except (EOFError, KeyboardInterrupt):
                text = None
            if text is None:
                self.emit("再见。")
                return

            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                if self.handle_command(text):
                    self.emit("再见。")
                    return
                continue

            try:
                task = self.ask(text)
            except AgentError as exc:
                self.emit(f"⚠️  {exc.message}")
                continue
            self.emit(self.render(task))

    def _llm_notice(self) -> str:
        if self.manager.settings.llm.use_mock_client():
            return (
                "当前使用 MockLLM（未配置 LLM_API_KEY）：链路完整，但回答是模板化的。\n"
                "想接真实模型：复制 .env.example 为 .env 并填写 LLM_API_KEY。"
            )
        profile = self.manager.settings.llm.primary
        return f"当前模型：{profile.model} @ {profile.base_url}"


def _line_reader(lines: Iterable[str] | None) -> Callable[[], str | None]:
    """返回一个"取下一行"的函数；取完或遇到 EOF 返回 None。"""
    if lines is None:
        return lambda: input("你 > ")
    iterator = iter(lines)
    return lambda: next(iterator, None)


def build_console(settings: AppSettings | None = None, *, verbose: bool = False) -> ConsoleAgent:
    used = settings or get_settings()
    return ConsoleAgent(build_task_manager(used), verbose=verbose)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="日志查询 Agent 控制台")
    parser.add_argument("--verbose", action="store_true", help="打印状态流转、步骤与工具调用")
    parser.add_argument("--query", help="只跑一条查询然后退出（便于脚本化验证）")
    args = parser.parse_args(argv)

    console = build_console(verbose=args.verbose)
    if args.query:
        console.emit(console.render(console.ask(args.query)))
        return 0
    console.run_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover - 入口
    raise SystemExit(main())
