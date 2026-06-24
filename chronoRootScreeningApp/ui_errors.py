"""Shared UI error handling and subprocess launching for the screening app."""

import os
import sys
from typing import Callable, List, Optional

from PyQt5.QtCore import QObject, QProcess
from PyQt5.QtWidgets import QMessageBox, QWidget


def show_warning(parent: Optional[QWidget], title: str, message: str) -> None:
    QMessageBox.warning(parent, title, message)


def show_critical(parent: Optional[QWidget], title: str, message: str) -> None:
    QMessageBox.critical(parent, title, message)


def show_information(parent: Optional[QWidget], title: str, message: str) -> None:
    QMessageBox.information(parent, title, message)


class WorkerLauncher(QObject):
    """Launch a worker process without shell=True and report failures via popups."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        working_directory: Optional[str] = None,
    ):
        super().__init__(parent)
        self._dialog_parent = parent
        self.working_directory = working_directory or os.path.dirname(os.path.abspath(__file__))
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.SeparateChannels)
        self.process.setWorkingDirectory(self.working_directory)
        self._stderr_chunks: List[bytes] = []
        self._on_success: Optional[Callable[[], None]] = None
        self._success_title: Optional[str] = None
        self._success_message: Optional[str] = None
        self._error_title = "Processing Error"
        self.process.readyReadStandardError.connect(self._collect_stderr)
        self.process.finished.connect(self._on_finished)

    def _collect_stderr(self) -> None:
        data = self.process.readAllStandardError()
        if data:
            self._stderr_chunks.append(bytes(data))

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if exit_code == 0 and exit_status == QProcess.NormalExit:
            if self._on_success:
                self._on_success()
            elif self._success_title and self._success_message:
                show_information(self._dialog_parent, self._success_title, self._success_message)
            return

        stderr = b"".join(self._stderr_chunks).decode("utf-8", errors="replace").strip()
        stdout = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
        details = stderr or stdout or f"Process exited with code {exit_code}."
        if len(details) > 2000:
            details = "...\n" + details[-2000:]
        show_critical(self._dialog_parent, self._error_title, details)

    def start(
        self,
        args: List[str],
        *,
        started_title: Optional[str] = None,
        started_message: Optional[str] = None,
        success_title: Optional[str] = None,
        success_message: Optional[str] = None,
        error_title: str = "Processing Error",
        on_success: Optional[Callable[[], None]] = None,
    ) -> bool:
        if self.process.state() != QProcess.NotRunning:
            show_warning(self._dialog_parent, "Busy", "A background task is already running.")
            return False

        self._stderr_chunks = []
        self._on_success = on_success
        self._success_title = success_title
        self._success_message = success_message
        self._error_title = error_title

        program = args[0]
        program_args = args[1:]
        if program == "python":
            program = sys.executable
            program_args = args[1:]

        self.process.start(program, program_args)
        if not self.process.waitForStarted(5000):
            show_critical(
                self._dialog_parent,
                error_title,
                f"Failed to start process:\n{self.process.errorString()}",
            )
            return False

        if started_title and started_message:
            show_information(self._dialog_parent, started_title, started_message)
        return True

    def is_running(self) -> bool:
        return self.process.state() != QProcess.NotRunning


def launch_worker(
    args: List[str],
    parent: Optional[QWidget] = None,
    *,
    started_title: Optional[str] = None,
    started_message: Optional[str] = None,
    success_title: Optional[str] = None,
    success_message: Optional[str] = None,
    error_title: str = "Processing Error",
    working_directory: Optional[str] = None,
    on_success: Optional[Callable[[], None]] = None,
) -> WorkerLauncher:
    launcher = WorkerLauncher(parent=parent, working_directory=working_directory)
    launcher.start(
        args,
        started_title=started_title,
        started_message=started_message,
        success_title=success_title,
        success_message=success_message,
        error_title=error_title,
        on_success=on_success,
    )
    return launcher
