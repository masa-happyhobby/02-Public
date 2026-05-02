import json
import queue
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import pyqtgraph as pg
import serial
import serial.rs485
from serial.tools import list_ports
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import QThread
from logger_config import setup_logger

# ロガー作成
logger = setup_logger(__name__)

APP_NAME = "Serial Debug Tool"
CONFIG_PATH = Path("serial_debug_tool_config.json")

DEFAULT_CONFIG: dict[str, Any] = {
    "serial_settings": {
        "port": "",
        "baudrate": 115200,
        "bytesize": 8,
        "parity": "N",
        "stopbits": 1,
        "timeout": 0.05,
        "write_timeout": 0.5,
        "flow_control": "None",
        "interface_type": "RS-232C",
        "rs485": {
            "enabled": False,
            "rts_level_for_tx": True,
            "rts_level_for_rx": False,
            "loopback": False,
            "delay_before_tx": 0.0,
            "delay_before_rx": 0.0,
        },
    },
    "commands": [
        {
            "name": "PING_TEXT",
            "payload_type": "TEXT",
            "payload": "PING\\r\\n",
            "encoding": "ascii",
            "expect_reply": True,
            "timeout_ms": 300,
            "terminator_hex": "0D0A",
        },
        {
            "name": "READ_STATUS_HEX",
            "payload_type": "HEX",
            "payload": "AA 55 01 00",
            "encoding": "ascii",
            "expect_reply": True,
            "timeout_ms": 300,
            "terminator_hex": "",
        },
    ],
    "sequences": [
        {
            "name": "BOOT_CHECK",
            "loop_count": 1,
            "steps_text": "PING_TEXT,2,100\nREAD_STATUS_HEX,1,100",
        }
    ],
    "self_test": {
        "enabled": False,
        "peer_port": "",
        "reply_mode": "echo",
        "reply_type": "TEXT",
        "reply_payload": "ACK\r\n",
        "periodic_type": "TEXT",
        "periodic_payload": "",
        "periodic_interval": 0.0,
        "encoding": "ascii",
    },
}


@dataclass
class TransactionRequest:
    request_id: str
    name: str
    payload: bytes
    expect_reply: bool
    timeout_ms: int
    terminator: Optional[bytes]
    sequence_name: str = ""
    step_index: int = -1
    retries_left: int = 0
    metadata: Optional[dict[str, Any]] = None


class CommandDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, data: Optional[dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle("コマンド編集")
        self.resize(520, 380)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.name_edit = QLineEdit()
        self.payload_type_combo = QComboBox()
        self.payload_type_combo.addItems(["TEXT", "HEX"])
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems(["ascii", "utf-8", "shift_jis"])
        self.expect_reply_check = QCheckBox("応答を待つ")
        self.expect_reply_check.setChecked(True)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1, 60000)
        self.timeout_spin.setSuffix(" ms")
        self.timeout_spin.setValue(300)
        self.terminator_edit = QLineEdit()
        self.terminator_edit.setPlaceholderText("例: 0D0A")
        self.payload_edit = QPlainTextEdit()
        self.payload_edit.setPlaceholderText("TEXTなら文字列、HEXなら'AA 55 01 00'のように入力")

        form.addRow("名前", self.name_edit)
        form.addRow("データ形式", self.payload_type_combo)
        form.addRow("文字コード", self.encoding_combo)
        form.addRow("応答待ち", self.expect_reply_check)
        form.addRow("応答タイムアウト", self.timeout_spin)
        form.addRow("終端HEX", self.terminator_edit)
        layout.addLayout(form)
        layout.addWidget(QLabel("電文"))
        layout.addWidget(self.payload_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if data:
            self.name_edit.setText(data.get("name", ""))
            self.payload_type_combo.setCurrentText(data.get("payload_type", "TEXT"))
            self.encoding_combo.setCurrentText(data.get("encoding", "ascii"))
            self.expect_reply_check.setChecked(bool(data.get("expect_reply", True)))
            self.timeout_spin.setValue(int(data.get("timeout_ms", 300)))
            self.terminator_edit.setText(data.get("terminator_hex", ""))
            self.payload_edit.setPlainText(data.get("payload", ""))

    def get_data(self) -> dict[str, Any]:
        return {
            "name": self.name_edit.text().strip(),
            "payload_type": self.payload_type_combo.currentText(),
            "payload": self.payload_edit.toPlainText(),
            "encoding": self.encoding_combo.currentText(),
            "expect_reply": self.expect_reply_check.isChecked(),
            "timeout_ms": self.timeout_spin.value(),
            "terminator_hex": self.terminator_edit.text().strip(),
        }


class SequenceDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, data: Optional[dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle("シーケンス編集")
        self.resize(560, 420)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.loop_spin = QSpinBox()
        self.loop_spin.setRange(1, 100000)
        self.loop_spin.setValue(1)
        form.addRow("名前", self.name_edit)
        form.addRow("ループ回数", self.loop_spin)
        layout.addLayout(form)

        helper = QLabel(
            "手順は1行1ステップで入力します。\n"
            "形式: コマンド名,再送回数,待機ms\n"
            "例: PING_TEXT,2,100"
        )
        helper.setWordWrap(True)
        layout.addWidget(helper)

        self.steps_edit = QPlainTextEdit()
        layout.addWidget(self.steps_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if data:
            self.name_edit.setText(data.get("name", ""))
            self.loop_spin.setValue(int(data.get("loop_count", 1)))
            self.steps_edit.setPlainText(data.get("steps_text", ""))

    def get_data(self) -> dict[str, Any]:
        return {
            "name": self.name_edit.text().strip(),
            "loop_count": self.loop_spin.value(),
            "steps_text": self.steps_edit.toPlainText().strip(),
        }


class SerialWorker(QThread):
    log_message = Signal(str)
    state_changed = Signal(str)
    transaction_finished = Signal(dict)
    spontaneous_received = Signal(bytes)
    line_state_changed = Signal(dict)
    connection_changed = Signal(bool, str)

    def __init__(self):
        super().__init__()
        self._cmd_queue: queue.Queue[TransactionRequest] = queue.Queue()
        self._open_settings: Optional[dict[str, Any]] = None
        self._close_requested = False
        self._lock = threading.Lock()
        self._running = True
        self._ser: Optional[serial.Serial] = None
        self._tx_pulse_until = 0.0
        self._rx_pulse_until = 0.0
        self._busy_until = 0.0

    def request_open(self, settings: dict[str, Any]) -> None:
        with self._lock:
            self._open_settings = dict(settings)
            self._close_requested = False

    def request_close(self) -> None:
        with self._lock:
            self._close_requested = True
            while not self._cmd_queue.empty():
                try:
                    self._cmd_queue.get_nowait()
                except queue.Empty:
                    break

    def submit_transaction(self, req: TransactionRequest) -> None:
        self._cmd_queue.put(req)

    def stop_thread(self) -> None:
        self._running = False
        self.request_close()
        self.wait(2000)

    def run(self) -> None:
        last_line_poll = 0.0

        while self._running:
            self._handle_open_close()

            if self._ser and self._ser.is_open:
                try:
                    if self._ser.in_waiting:
                        data = self._ser.read(self._ser.in_waiting)
                        if data:
                            self._rx_pulse_until = time.perf_counter() + 0.25
                            self.log_message.emit(f"受信(自然流入) HEX: {bytes_to_hex(data)}")
                            self.spontaneous_received.emit(bytes(data))
                except Exception as ex:
                    self.log_message.emit(f"受信エラー: {ex}")
                    self._safe_close()
                    continue

                try:
                    req = self._cmd_queue.get_nowait()
                    self._execute_transaction(req)
                except queue.Empty:
                    pass

                now = time.perf_counter()
                if now - last_line_poll >= 0.1:
                    self.line_state_changed.emit(self._current_line_state())
                    last_line_poll = now
            else:
                time.sleep(0.05)

    def _handle_open_close(self) -> None:
        open_settings: Optional[dict[str, Any]] = None
        close_requested = False
        with self._lock:
            if self._open_settings is not None:
                open_settings = self._open_settings
                self._open_settings = None
            close_requested = self._close_requested
            self._close_requested = False

        if close_requested:
            self._safe_close()

        if open_settings is not None:
            self._safe_close()
            self._open_serial(open_settings)

    def _open_serial(self, settings: dict[str, Any]) -> None:
        try:
            self.state_changed.emit("接続中")
            self.log_message.emit(f"ポートを開きます: {settings['port']}")

            flow = settings.get("flow_control", "None")
            rtscts = flow == "RTS/CTS"
            dsrdtr = flow == "DSR/DTR"
            xonxoff = flow == "XON/XOFF"

            ser = serial.Serial(
                port=settings["port"],
                baudrate=int(settings["baudrate"]),
                bytesize=map_bytesize(settings.get("bytesize", 8)),
                parity=map_parity(settings.get("parity", "N")),
                stopbits=map_stopbits(settings.get("stopbits", 1)),
                timeout=float(settings.get("timeout", 0.05)),
                write_timeout=float(settings.get("write_timeout", 0.5)),
                rtscts=rtscts,
                dsrdtr=dsrdtr,
                xonxoff=xonxoff,
            )

            interface_type = settings.get("interface_type", "RS-232C")
            if interface_type == "RS-485":
                rs485_settings = settings.get("rs485", {})
                if rs485_settings.get("enabled", True):
                    ser.rs485_mode = serial.rs485.RS485Settings(
                        rts_level_for_tx=bool(rs485_settings.get("rts_level_for_tx", True)),
                        rts_level_for_rx=bool(rs485_settings.get("rts_level_for_rx", False)),
                        loopback=bool(rs485_settings.get("loopback", False)),
                        delay_before_tx=float(rs485_settings.get("delay_before_tx", 0.0)),
                        delay_before_rx=float(rs485_settings.get("delay_before_rx", 0.0)),
                    )
                    self.log_message.emit("RS-485設定を適用しました。")
            elif interface_type == "RS-422A":
                self.log_message.emit("RS-422Aは接続アダプタ側の物理対応が必要です。ソフト側は通常シリアル設定で開いています。")
            else:
                self.log_message.emit("RS-232Cプロファイルで開きました。")

            self._ser = ser
            self.state_changed.emit("待機")
            self.connection_changed.emit(True, ser.port)
            self.log_message.emit("接続成功")
        except Exception as ex:
            self._ser = None
            self.state_changed.emit("切断")
            self.connection_changed.emit(False, "")
            self.log_message.emit(f"接続失敗: {ex}")

    def _safe_close(self) -> None:
        if self._ser:
            try:
                port_name = self._ser.port
            except Exception:
                port_name = ""
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
            self.state_changed.emit("切断")
            self.connection_changed.emit(False, "")
            self.log_message.emit(f"切断しました: {port_name}")

    def _execute_transaction(self, req: TransactionRequest) -> None:
        if not self._ser or not self._ser.is_open:
            self.transaction_finished.emit(
                {
                    "request_id": req.request_id,
                    "ok": False,
                    "name": req.name,
                    "tx_hex": bytes_to_hex(req.payload),
                    "rx_hex": "",
                    "elapsed_ms": 0.0,
                    "message": "未接続です",
                    "sequence_name": req.sequence_name,
                    "step_index": req.step_index,
                    "retries_left": req.retries_left,
                }
            )
            return

        started = time.perf_counter()
        rx = bytearray()
        ok = False
        message = ""

        try:
            self.state_changed.emit("送信中")
            self._busy_until = time.perf_counter() + 0.3
            self._tx_pulse_until = time.perf_counter() + 0.25
            self.log_message.emit(f"送信[{req.name}] HEX: {bytes_to_hex(req.payload)}")
            self._ser.reset_input_buffer()
            self._ser.write(req.payload)
            self._ser.flush()

            if req.expect_reply:
                self.state_changed.emit("受信中")
                deadline = time.perf_counter() + (req.timeout_ms / 1000.0)
                while time.perf_counter() < deadline:
                    n = self._ser.in_waiting
                    if n:
                        chunk = self._ser.read(n)
                        if chunk:
                            rx.extend(chunk)
                            self._busy_until = time.perf_counter() + 0.3
                            self._rx_pulse_until = time.perf_counter() + 0.25
                            if req.terminator and req.terminator in rx:
                                ok = True
                                message = "終端を検出しました"
                                break
                    else:
                        time.sleep(0.005)

                if not ok:
                    if len(rx) > 0:
                        ok = True
                        message = "応答を受信しました"
                    else:
                        message = "応答タイムアウト"
            else:
                ok = True
                message = "送信完了"

        except Exception as ex:
            message = f"通信例外: {ex}"
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result = {
                "request_id": req.request_id,
                "ok": ok,
                "name": req.name,
                "tx_hex": bytes_to_hex(req.payload),
                "tx_ascii": safe_ascii(req.payload),
                "rx_hex": bytes_to_hex(rx),
                "rx_ascii": safe_ascii(rx),
                "elapsed_ms": round(elapsed_ms, 3),
                "message": message,
                "sequence_name": req.sequence_name,
                "step_index": req.step_index,
                "retries_left": req.retries_left,
            }
            self.transaction_finished.emit(result)
            self.log_message.emit(
                f"完了[{req.name}] ok={ok} time={elapsed_ms:.3f}ms rx={bytes_to_hex(rx) if rx else '(none)'}"
            )
            self.state_changed.emit("待機" if self._ser and self._ser.is_open else "切断")

    def _current_line_state(self) -> dict[str, int]:
        now = time.perf_counter()
        tx = 1 if now < self._tx_pulse_until else 0
        rx = 1 if now < self._rx_pulse_until else 0
        busy = 1 if now < self._busy_until else 0
        state = {
            "TX": tx,
            "RX": rx,
            "BUSY": busy,
            "RTS": 0,
            "CTS": 0,
            "DSR": 0,
            "DCD": 0,
            "RI": 0,
        }
        if not self._ser or not self._ser.is_open:
            return state
        try:
            state["RTS"] = 1 if bool(self._ser.rts) else 0
        except Exception:
            pass
        try:
            state["CTS"] = 1 if bool(self._ser.getCTS()) else 0
        except Exception:
            pass
        try:
            state["DSR"] = 1 if bool(self._ser.getDSR()) else 0
        except Exception:
            pass
        try:
            state["DCD"] = 1 if bool(self._ser.getCD()) else 0
        except Exception:
            pass
        try:
            state["RI"] = 1 if bool(self._ser.getRI()) else 0
        except Exception:
            pass
        return state


class SelfTestPeerThread(QThread):
    log_message = Signal(str)
    running_changed = Signal(bool, str)

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._open_settings: Optional[dict[str, Any]] = None
        self._close_requested = False
        self._running = True
        self._ser: Optional[serial.Serial] = None
        self._next_periodic_at: Optional[float] = None
        self._active_settings: dict[str, Any] = {}

    def request_start(self, settings: dict[str, Any]) -> None:
        with self._lock:
            self._open_settings = dict(settings)
            self._close_requested = False

    def request_stop(self) -> None:
        with self._lock:
            self._close_requested = True

    def stop_thread(self) -> None:
        self._running = False
        self.request_stop()
        self.wait(2000)

    def run(self) -> None:
        while self._running:
            self._handle_open_close()

            if not self._ser or not self._ser.is_open:
                time.sleep(0.05)
                continue

            try:
                self._handle_periodic_send()

                waiting = self._ser.in_waiting
                if waiting:
                    data = self._ser.read(waiting)
                    if data:
                        self.log_message.emit(
                            f"[SELFTEST] 受信 HEX={bytes_to_hex(data)} ASCII={safe_ascii(data)}"
                        )
                        reply = self._build_reply(data)
                        if reply:
                            started = time.perf_counter()
                            self._ser.write(reply)
                            self._ser.flush()
                            elapsed_ms = (time.perf_counter() - started) * 1000.0
                            self.log_message.emit(
                                f"[SELFTEST] 返信 HEX={bytes_to_hex(reply)} ASCII={safe_ascii(reply)} time={elapsed_ms:.3f}ms"
                            )
                else:
                    time.sleep(0.005)
            except Exception as ex:
                self.log_message.emit(f"[SELFTEST] 例外: {ex}")
                self._safe_close()

    def _handle_open_close(self) -> None:
        open_settings: Optional[dict[str, Any]] = None
        close_requested = False
        with self._lock:
            if self._open_settings is not None:
                open_settings = self._open_settings
                self._open_settings = None
            close_requested = self._close_requested
            self._close_requested = False

        if close_requested:
            self._safe_close()

        if open_settings is not None:
            self._safe_close()
            self._open_serial(open_settings)

    def _open_serial(self, settings: dict[str, Any]) -> None:
        try:
            port = settings.get('peer_port', '').strip()
            if not port:
                raise ValueError('自己試験用の相手側ポートが未入力です。')

            ser = serial.Serial(
                port=port,
                baudrate=int(settings['baudrate']),
                bytesize=map_bytesize(settings.get('bytesize', 8)),
                parity=map_parity(settings.get('parity', 'N')),
                stopbits=map_stopbits(settings.get('stopbits', 1)),
                timeout=float(settings.get('timeout', 0.05)),
                write_timeout=float(settings.get('write_timeout', 0.5)),
                rtscts=settings.get('flow_control', 'None') == 'RTS/CTS',
                dsrdtr=settings.get('flow_control', 'None') == 'DSR/DTR',
                xonxoff=settings.get('flow_control', 'None') == 'XON/XOFF',
            )

            self._ser = ser
            self._active_settings = dict(settings)
            periodic_interval = float(settings.get('periodic_interval', 0.0))
            periodic_payload = parse_payload(
                settings.get('periodic_type', 'TEXT'),
                settings.get('periodic_payload', ''),
                settings.get('encoding', 'ascii'),
            ) if settings.get('periodic_payload', '') else b''
            self._active_settings['_periodic_payload_bytes'] = periodic_payload
            self._next_periodic_at = (time.perf_counter() + periodic_interval) if periodic_payload and periodic_interval > 0 else None
            self.log_message.emit(f"[SELFTEST] 相手側ポートを開きました: {port}")
            self.running_changed.emit(True, port)
        except Exception as ex:
            self._ser = None
            self._active_settings = {}
            self._next_periodic_at = None
            self.log_message.emit(f"[SELFTEST] 起動失敗: {ex}")
            self.running_changed.emit(False, '')

    def _safe_close(self) -> None:
        if self._ser:
            port = ''
            try:
                port = self._ser.port
            except Exception:
                pass
            try:
                self._ser.close()
            except Exception:
                pass
            self.log_message.emit(f"[SELFTEST] 相手側ポートを閉じました: {port}")
        self._ser = None
        self._active_settings = {}
        self._next_periodic_at = None
        self.running_changed.emit(False, '')

    def _build_reply(self, request_data: bytes) -> bytes:
        mode = str(self._active_settings.get('reply_mode', 'echo')).lower()
        if mode == 'none':
            return b''
        if mode == 'fixed':
            return parse_payload(
                self._active_settings.get('reply_type', 'TEXT'),
                self._active_settings.get('reply_payload', ''),
                self._active_settings.get('encoding', 'ascii'),
            )
        return bytes(request_data)

    def _handle_periodic_send(self) -> None:
        if not self._ser or not self._ser.is_open:
            return
        periodic_payload = self._active_settings.get('_periodic_payload_bytes', b'')
        periodic_interval = float(self._active_settings.get('periodic_interval', 0.0))
        if not periodic_payload or periodic_interval <= 0 or self._next_periodic_at is None:
            return
        now = time.perf_counter()
        if now < self._next_periodic_at:
            return
        started = time.perf_counter()
        self._ser.write(periodic_payload)
        self._ser.flush()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.log_message.emit(
            f"[SELFTEST] 周期送信 HEX={bytes_to_hex(periodic_payload)} ASCII={safe_ascii(periodic_payload)} time={elapsed_ms:.3f}ms"
        )
        self._next_periodic_at = now + periodic_interval



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Serial Debug Tool")
        self.resize(1500, 900)
        self.config = load_config()
        self.worker = SerialWorker()
        self.worker.start()
        self.self_test_peer = SelfTestPeerThread()
        self.self_test_peer.start()

        self.commands: list[dict[str, Any]] = list(self.config.get("commands", []))
        self.sequences: list[dict[str, Any]] = list(self.config.get("sequences", []))
        self.pending_requests: dict[str, dict[str, Any]] = {}
        self.sequence_context: Optional[dict[str, Any]] = None

        self.signal_names = ["TX", "RX", "BUSY", "RTS", "CTS", "DSR", "DCD", "RI"]
        self.signal_current = {name: 0 for name in self.signal_names}
        self.signal_history = {name: [] for name in self.signal_names}
        self.signal_time: list[float] = []
        self.signal_curves = {}
        self.plot_connection_start: Optional[float] = None
        self.max_plot_points = 20000

        self._build_ui()
        self._connect_signals()
        self._load_config_into_ui()
        self.refresh_ports()
        self.refresh_command_table()
        self.refresh_sequence_table()

        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self._update_plot)
        self.plot_timer.start(100)

    def closeEvent(self, event):
        self._save_all()
        self.worker.stop_thread()
        self.self_test_peer.stop_thread()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        root.addWidget(self._build_connection_group())

        splitter = QSplitter(Qt.Orientation.Vertical)
        upper = QSplitter(Qt.Orientation.Horizontal)
        upper.addWidget(self._build_command_group())
        upper.addWidget(self._build_sequence_group())
        upper.setStretchFactor(0, 1)
        upper.setStretchFactor(1, 1)

        lower = QSplitter(Qt.Orientation.Horizontal)
        lower.addWidget(self._build_plot_group())
        lower.addWidget(self._build_log_group())
        lower.setStretchFactor(0, 1)
        lower.setStretchFactor(1, 1)

        splitter.addWidget(upper)
        splitter.addWidget(lower)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        self._build_menu()

        status = QStatusBar()
        self.setStatusBar(status)
        self.lbl_conn = QLabel("未接続")
        self.lbl_state = QLabel("状態: 切断")
        self.lbl_iface = QLabel("I/F: -")
        self.lbl_last_time = QLabel("通信時間: -")
        self.lbl_selftest = QLabel("自己試験: 停止")
        status.addPermanentWidget(self.lbl_conn)
        status.addPermanentWidget(self.lbl_state)
        status.addPermanentWidget(self.lbl_iface)
        status.addPermanentWidget(self.lbl_last_time)
        status.addPermanentWidget(self.lbl_selftest)

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("ファイル")
        act_save = QAction("設定を保存", self)
        act_save.triggered.connect(self._save_all)
        menu.addAction(act_save)

        act_export = QAction("設定を別名保存", self)
        act_export.triggered.connect(self.export_config)
        menu.addAction(act_export)

        act_import = QAction("設定を読み込む", self)
        act_import.triggered.connect(self.import_config)
        menu.addAction(act_import)

    def _build_connection_group(self) -> QWidget:
        box = QGroupBox("接続設定")
        layout = QGridLayout(box)

        self.cmb_port = QComboBox()
        self.cmb_port.setEditable(True)
        self.btn_refresh_ports = QPushButton("ポート更新")
        self.cmb_interface = QComboBox()
        self.cmb_interface.addItems(["RS-232C", "RS-485", "RS-422A"])
        self.spn_baud = QSpinBox()
        self.spn_baud.setRange(300, 10000000)
        self.spn_baud.setValue(115200)
        self.cmb_bytesize = QComboBox()
        self.cmb_bytesize.addItems(["5", "6", "7", "8"])
        self.cmb_parity = QComboBox()
        self.cmb_parity.addItems(["N", "E", "O", "M", "S"])
        self.cmb_stopbits = QComboBox()
        self.cmb_stopbits.addItems(["1", "1.5", "2"])
        self.cmb_flow = QComboBox()
        self.cmb_flow.addItems(["None", "RTS/CTS", "DSR/DTR", "XON/XOFF"])
        self.spn_timeout = QDoubleSpinBox()
        self.spn_timeout.setDecimals(3)
        self.spn_timeout.setRange(0.001, 60.0)
        self.spn_timeout.setSingleStep(0.01)
        self.spn_timeout.setValue(0.05)
        self.spn_write_timeout = QDoubleSpinBox()
        self.spn_write_timeout.setDecimals(3)
        self.spn_write_timeout.setRange(0.001, 60.0)
        self.spn_write_timeout.setSingleStep(0.01)
        self.spn_write_timeout.setValue(0.5)

        self.chk_rs485_enabled = QCheckBox("RS-485制御を有効")
        self.chk_rs485_enabled.setChecked(False)
        self.chk_rts_tx = QCheckBox("送信時RTS=ON")
        self.chk_rts_tx.setChecked(True)
        self.chk_rts_rx = QCheckBox("受信時RTS=ON")
        self.chk_loopback = QCheckBox("RS-485ループバック")
        self.spn_delay_before_tx = QDoubleSpinBox()
        self.spn_delay_before_tx.setDecimals(4)
        self.spn_delay_before_tx.setRange(0.0, 5.0)
        self.spn_delay_before_rx = QDoubleSpinBox()
        self.spn_delay_before_rx.setDecimals(4)
        self.spn_delay_before_rx.setRange(0.0, 5.0)

        self.btn_connect = QPushButton("接続")
        self.btn_disconnect = QPushButton("切断")
        self.btn_disconnect.setEnabled(False)

        row = 0
        layout.addWidget(QLabel("ポート"), row, 0)
        layout.addWidget(self.cmb_port, row, 1)
        layout.addWidget(self.btn_refresh_ports, row, 2)
        layout.addWidget(QLabel("インターフェイス"), row, 3)
        layout.addWidget(self.cmb_interface, row, 4)
        layout.addWidget(self.btn_connect, row, 5)
        layout.addWidget(self.btn_disconnect, row, 6)

        row += 1
        layout.addWidget(QLabel("Baud"), row, 0)
        layout.addWidget(self.spn_baud, row, 1)
        layout.addWidget(QLabel("Data bits"), row, 2)
        layout.addWidget(self.cmb_bytesize, row, 3)
        layout.addWidget(QLabel("Parity"), row, 4)
        layout.addWidget(self.cmb_parity, row, 5)
        layout.addWidget(QLabel("Stop bits"), row, 6)
        layout.addWidget(self.cmb_stopbits, row, 7)

        row += 1
        layout.addWidget(QLabel("Flow control"), row, 0)
        layout.addWidget(self.cmb_flow, row, 1)
        layout.addWidget(QLabel("Read timeout[s]"), row, 2)
        layout.addWidget(self.spn_timeout, row, 3)
        layout.addWidget(QLabel("Write timeout[s]"), row, 4)
        layout.addWidget(self.spn_write_timeout, row, 5)

        row += 1
        layout.addWidget(self.chk_rs485_enabled, row, 0, 1, 2)
        layout.addWidget(self.chk_rts_tx, row, 2, 1, 2)
        layout.addWidget(self.chk_rts_rx, row, 4, 1, 2)
        layout.addWidget(self.chk_loopback, row, 6, 1, 2)

        row += 1
        layout.addWidget(QLabel("TX前遅延[s]"), row, 0)
        layout.addWidget(self.spn_delay_before_tx, row, 1)
        layout.addWidget(QLabel("RX前遅延[s]"), row, 2)
        layout.addWidget(self.spn_delay_before_rx, row, 3)

        self.edt_selftest_peer_port = QLineEdit()
        self.edt_selftest_peer_port.setPlaceholderText("例: COM12")
        self.cmb_selftest_reply_mode = QComboBox()
        self.cmb_selftest_reply_mode.addItems(["echo", "fixed", "none"])
        self.cmb_selftest_reply_type = QComboBox()
        self.cmb_selftest_reply_type.addItems(["TEXT", "HEX"])
        self.edt_selftest_reply_payload = QLineEdit()
        self.edt_selftest_reply_payload.setPlaceholderText("固定応答。例: OK\r\n / AA 55")
        self.cmb_selftest_periodic_type = QComboBox()
        self.cmb_selftest_periodic_type.addItems(["TEXT", "HEX"])
        self.edt_selftest_periodic_payload = QLineEdit()
        self.edt_selftest_periodic_payload.setPlaceholderText("周期送信データ。空なら無効")
        self.spn_selftest_periodic_interval = QDoubleSpinBox()
        self.spn_selftest_periodic_interval.setDecimals(3)
        self.spn_selftest_periodic_interval.setRange(0.0, 3600.0)
        self.spn_selftest_periodic_interval.setSingleStep(0.1)
        self.btn_selftest_toggle = QPushButton("com0com自己試験開始")

        row += 1
        layout.addWidget(QLabel("自己試験 相手側ポート"), row, 0)
        layout.addWidget(self.edt_selftest_peer_port, row, 1)
        layout.addWidget(QLabel("応答モード"), row, 2)
        layout.addWidget(self.cmb_selftest_reply_mode, row, 3)
        layout.addWidget(self.btn_selftest_toggle, row, 5, 1, 2)

        row += 1
        layout.addWidget(QLabel("固定応答形式"), row, 0)
        layout.addWidget(self.cmb_selftest_reply_type, row, 1)
        layout.addWidget(QLabel("固定応答データ"), row, 2)
        layout.addWidget(self.edt_selftest_reply_payload, row, 3, 1, 3)

        row += 1
        layout.addWidget(QLabel("周期送信形式"), row, 0)
        layout.addWidget(self.cmb_selftest_periodic_type, row, 1)
        layout.addWidget(QLabel("周期送信データ"), row, 2)
        layout.addWidget(self.edt_selftest_periodic_payload, row, 3)
        layout.addWidget(QLabel("周期[s]"), row, 4)
        layout.addWidget(self.spn_selftest_periodic_interval, row, 5)

        return box

    def _build_command_group(self) -> QWidget:
        box = QGroupBox("電文定義")
        layout = QVBoxLayout(box)

        self.tbl_commands = QTableWidget(0, 6)
        self.tbl_commands.setHorizontalHeaderLabels(["名前", "形式", "電文", "応答待ち", "Timeout", "終端HEX"])
        self.tbl_commands.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.tbl_commands)

        buttons = QHBoxLayout()
        self.btn_add_cmd = QPushButton("追加")
        self.btn_edit_cmd = QPushButton("編集")
        self.btn_del_cmd = QPushButton("削除")
        self.btn_send_cmd = QPushButton("送信")
        buttons.addWidget(self.btn_add_cmd)
        buttons.addWidget(self.btn_edit_cmd)
        buttons.addWidget(self.btn_del_cmd)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_send_cmd)
        layout.addLayout(buttons)
        return box

    def _build_sequence_group(self) -> QWidget:
        box = QGroupBox("シーケンス定義")
        layout = QVBoxLayout(box)

        self.tbl_sequences = QTableWidget(0, 3)
        self.tbl_sequences.setHorizontalHeaderLabels(["名前", "Loop", "ステップ数"])
        self.tbl_sequences.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.tbl_sequences)

        buttons = QHBoxLayout()
        self.btn_add_seq = QPushButton("追加")
        self.btn_edit_seq = QPushButton("編集")
        self.btn_del_seq = QPushButton("削除")
        self.btn_run_seq = QPushButton("実行")
        self.btn_stop_seq = QPushButton("停止")
        buttons.addWidget(self.btn_add_seq)
        buttons.addWidget(self.btn_edit_seq)
        buttons.addWidget(self.btn_del_seq)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_run_seq)
        buttons.addWidget(self.btn_stop_seq)
        layout.addLayout(buttons)
        return box

    def _build_log_group(self) -> QWidget:
        box = QGroupBox("ログ")
        layout = QVBoxLayout(box)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        layout.addWidget(self.txt_log)
        return box

    def _build_plot_group(self) -> QWidget:
        box = QGroupBox("リアルタイム信号")
        layout = QVBoxLayout(box)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0f172a")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "time", units="s")
        self.plot.setYRange(-1, len(self.signal_names) * 2 + 1)
        self.plot.addLegend(offset=(10, 10))
        colors = ["#60a5fa", "#34d399", "#f59e0b", "#a78bfa", "#f472b6", "#f87171", "#22d3ee", "#eab308"]
        for idx, name in enumerate(self.signal_names):
            pen = pg.mkPen(colors[idx], width=2)
            self.signal_curves[name] = self.plot.plot([], [], pen=pen, name=name)
        layout.addWidget(self.plot)

        self.lst_line_state = QListWidget()
        for name in self.signal_names:
            QListWidgetItem(f"{name}: 0", self.lst_line_state)
        layout.addWidget(self.lst_line_state)
        return box

    def _connect_signals(self) -> None:
        self.btn_refresh_ports.clicked.connect(self.refresh_ports)
        self.btn_connect.clicked.connect(self.connect_serial)
        self.btn_disconnect.clicked.connect(self.disconnect_serial)
        self.btn_selftest_toggle.clicked.connect(self.toggle_self_test)
        self.btn_add_cmd.clicked.connect(self.add_command)
        self.btn_edit_cmd.clicked.connect(self.edit_command)
        self.btn_del_cmd.clicked.connect(self.delete_command)
        self.btn_send_cmd.clicked.connect(self.send_selected_command)
        self.btn_add_seq.clicked.connect(self.add_sequence)
        self.btn_edit_seq.clicked.connect(self.edit_sequence)
        self.btn_del_seq.clicked.connect(self.delete_sequence)
        self.btn_run_seq.clicked.connect(self.run_selected_sequence)
        self.btn_stop_seq.clicked.connect(self.stop_sequence)

        self.worker.log_message.connect(self.append_log)
        self.worker.state_changed.connect(self.on_state_changed)
        self.worker.transaction_finished.connect(self.on_transaction_finished)
        self.worker.spontaneous_received.connect(self.on_spontaneous_received)
        self.worker.line_state_changed.connect(self.on_line_state_changed)
        self.worker.connection_changed.connect(self.on_connection_changed)
        self.self_test_peer.log_message.connect(self.append_log)
        self.self_test_peer.running_changed.connect(self.on_self_test_running_changed)

    def _load_config_into_ui(self) -> None:
        settings = self.config.get("serial_settings", DEFAULT_CONFIG["serial_settings"])
        self.spn_baud.setValue(int(settings.get("baudrate", 115200)))
        self.cmb_bytesize.setCurrentText(str(settings.get("bytesize", 8)))
        self.cmb_parity.setCurrentText(str(settings.get("parity", "N")))
        self.cmb_stopbits.setCurrentText(str(settings.get("stopbits", 1)))
        self.cmb_flow.setCurrentText(str(settings.get("flow_control", "None")))
        self.cmb_interface.setCurrentText(str(settings.get("interface_type", "RS-232C")))
        self.spn_timeout.setValue(float(settings.get("timeout", 0.05)))
        self.spn_write_timeout.setValue(float(settings.get("write_timeout", 0.5)))
        rs485 = settings.get("rs485", {})
        self.chk_rs485_enabled.setChecked(bool(rs485.get("enabled", False)))
        self.chk_rts_tx.setChecked(bool(rs485.get("rts_level_for_tx", True)))
        self.chk_rts_rx.setChecked(bool(rs485.get("rts_level_for_rx", False)))
        self.chk_loopback.setChecked(bool(rs485.get("loopback", False)))
        self.spn_delay_before_tx.setValue(float(rs485.get("delay_before_tx", 0.0)))
        self.spn_delay_before_rx.setValue(float(rs485.get("delay_before_rx", 0.0)))
        self.lbl_iface.setText(f"I/F: {self.cmb_interface.currentText()}")
        self_test = self.config.get("self_test", DEFAULT_CONFIG["self_test"])
        self.edt_selftest_peer_port.setText(str(self_test.get("peer_port", "")))
        self.cmb_selftest_reply_mode.setCurrentText(str(self_test.get("reply_mode", "echo")))
        self.cmb_selftest_reply_type.setCurrentText(str(self_test.get("reply_type", "TEXT")))
        self.edt_selftest_reply_payload.setText(str(self_test.get("reply_payload", "ACK\r\n")))
        self.cmb_selftest_periodic_type.setCurrentText(str(self_test.get("periodic_type", "TEXT")))
        self.edt_selftest_periodic_payload.setText(str(self_test.get("periodic_payload", "")))
        self.spn_selftest_periodic_interval.setValue(float(self_test.get("periodic_interval", 0.0)))

    def collect_serial_settings(self) -> dict[str, Any]:
        return {
            "port": self.cmb_port.currentText(),
            "baudrate": self.spn_baud.value(),
            "bytesize": int(self.cmb_bytesize.currentText()),
            "parity": self.cmb_parity.currentText(),
            "stopbits": float(self.cmb_stopbits.currentText()),
            "timeout": self.spn_timeout.value(),
            "write_timeout": self.spn_write_timeout.value(),
            "flow_control": self.cmb_flow.currentText(),
            "interface_type": self.cmb_interface.currentText(),
            "rs485": {
                "enabled": self.chk_rs485_enabled.isChecked(),
                "rts_level_for_tx": self.chk_rts_tx.isChecked(),
                "rts_level_for_rx": self.chk_rts_rx.isChecked(),
                "loopback": self.chk_loopback.isChecked(),
                "delay_before_tx": self.spn_delay_before_tx.value(),
                "delay_before_rx": self.spn_delay_before_rx.value(),
            },
        }

    def collect_self_test_settings(self) -> dict[str, Any]:
        serial_settings = self.collect_serial_settings()
        return {
            "peer_port": self.edt_selftest_peer_port.text().strip(),
            "reply_mode": self.cmb_selftest_reply_mode.currentText(),
            "reply_type": self.cmb_selftest_reply_type.currentText(),
            "reply_payload": self.edt_selftest_reply_payload.text(),
            "periodic_type": self.cmb_selftest_periodic_type.currentText(),
            "periodic_payload": self.edt_selftest_periodic_payload.text(),
            "periodic_interval": self.spn_selftest_periodic_interval.value(),
            "encoding": "ascii",
            "baudrate": serial_settings["baudrate"],
            "bytesize": serial_settings["bytesize"],
            "parity": serial_settings["parity"],
            "stopbits": serial_settings["stopbits"],
            "timeout": serial_settings["timeout"],
            "write_timeout": serial_settings["write_timeout"],
            "flow_control": serial_settings["flow_control"],
        }

    def toggle_self_test(self) -> None:
        if self.btn_selftest_toggle.text().endswith("停止"):
            self.self_test_peer.request_stop()
            return

        settings = self.collect_self_test_settings()
        if not settings["peer_port"]:
            QMessageBox.warning(self, "確認", "自己試験用の相手側ポートを入力してください。")
            return

        main_port = self.cmb_port.currentText().strip()
        if main_port and settings["peer_port"].upper() == main_port.upper():
            QMessageBox.warning(self, "確認", "メインポートと相手側ポートは別にしてください。")
            return

        self.append_log(
            f"自己試験を開始します。main={main_port or '(未入力)'} peer={settings['peer_port']} mode={settings['reply_mode']}"
        )
        self.self_test_peer.request_start(settings)

    def on_self_test_running_changed(self, running: bool, port_name: str) -> None:
        if running:
            self.btn_selftest_toggle.setText("com0com自己試験停止")
            self.lbl_selftest.setText(f"自己試験: 動作中({port_name})")
        else:
            self.btn_selftest_toggle.setText("com0com自己試験開始")
            self.lbl_selftest.setText("自己試験: 停止")

    def refresh_ports(self) -> None:
        current = self.cmb_port.currentText().strip()
        saved = str(self.config.get("serial_settings", {}).get("port", "")).strip()
        self.cmb_port.clear()
        ports = sorted(list_ports.comports(), key=lambda p: p.device)
        seen: set[str] = set()
        for p in ports:
            self.cmb_port.addItem(p.device)
            seen.add(p.device)

        for extra in [current, saved]:
            if extra and extra not in seen:
                self.cmb_port.addItem(extra)
                seen.add(extra)

        target = current or saved
        if target:
            idx = self.cmb_port.findText(target)
            if idx >= 0:
                self.cmb_port.setCurrentIndex(idx)
            else:
                self.cmb_port.setEditText(target)

    def connect_serial(self) -> None:
        port = self.cmb_port.currentText().strip()
        if not port:
            QMessageBox.warning(self, "確認", "シリアルポートを選択してください。")
            return
        settings = self.collect_serial_settings()
        self.worker.request_open(settings)
        self.lbl_iface.setText(f"I/F: {settings['interface_type']}")

    def disconnect_serial(self) -> None:
        self.worker.request_close()

    def on_connection_changed(self, connected: bool, port_name: str) -> None:
        self.btn_connect.setEnabled(not connected)
        self.btn_disconnect.setEnabled(connected)
        self.lbl_conn.setText(f"接続: {port_name if connected else '未接続'}")

        if connected:
            self.plot_connection_start = time.perf_counter()
            self.signal_time.clear()
            for name in self.signal_names:
                self.signal_history[name].clear()
            self.plot.enableAutoRange(axis='x', enable=True)
            self.plot.setXRange(0, 1, padding=0)
        else:
            self.plot_connection_start = None

    def on_state_changed(self, state: str) -> None:
        self.lbl_state.setText(f"状態: {state}")

    def append_log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.txt_log.append(f"[{ts}] {message}")

    def on_spontaneous_received(self, data: bytes) -> None:
        self.append_log(f"受信ASCII: {safe_ascii(data)}")

    def on_line_state_changed(self, state: dict) -> None:
        for idx, name in enumerate(self.signal_names):
            value = int(state.get(name, 0))
            self.signal_current[name] = value
            self.lst_line_state.item(idx).setText(f"{name}: {value}")

    def _update_plot(self) -> None:
        if self.plot_connection_start is None:
            return

        elapsed = time.perf_counter() - self.plot_connection_start
        self.signal_time.append(elapsed)

        for name in self.signal_names:
            self.signal_history[name].append(self.signal_current[name])

        if len(self.signal_time) > self.max_plot_points:
            self.signal_time.pop(0)
            for name in self.signal_names:
                if self.signal_history[name]:
                    self.signal_history[name].pop(0)

        if not self.signal_time:
            return

        x = self.signal_time
        for idx, name in enumerate(self.signal_names):
            base = idx * 2
            y = [base + v for v in self.signal_history[name]]
            self.signal_curves[name].setData(x, y)

        latest_x = x[-1]
        self.plot.setXRange(0, max(1.0, latest_x), padding=0.02)

    def refresh_command_table(self) -> None:
        self.tbl_commands.setRowCount(len(self.commands))
        for row, cmd in enumerate(self.commands):
            values = [
                cmd.get("name", ""),
                cmd.get("payload_type", ""),
                cmd.get("payload", "").replace("\n", "\\n"),
                "Yes" if cmd.get("expect_reply", True) else "No",
                str(cmd.get("timeout_ms", 0)),
                cmd.get("terminator_hex", ""),
            ]
            for col, val in enumerate(values):
                self.tbl_commands.setItem(row, col, QTableWidgetItem(val))
        self.tbl_commands.resizeColumnsToContents()

    def refresh_sequence_table(self) -> None:
        self.tbl_sequences.setRowCount(len(self.sequences))
        for row, seq in enumerate(self.sequences):
            steps = parse_steps_text(seq.get("steps_text", ""))
            values = [seq.get("name", ""), str(seq.get("loop_count", 1)), str(len(steps))]
            for col, val in enumerate(values):
                self.tbl_sequences.setItem(row, col, QTableWidgetItem(val))
        self.tbl_sequences.resizeColumnsToContents()

    def selected_command_index(self) -> int:
        row = self.tbl_commands.currentRow()
        return row

    def selected_sequence_index(self) -> int:
        row = self.tbl_sequences.currentRow()
        return row

    def add_command(self) -> None:
        dlg = CommandDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if not data["name"]:
                QMessageBox.warning(self, "確認", "名前を入力してください。")
                return
            self.commands.append(data)
            self.refresh_command_table()
            self._save_all()

    def edit_command(self) -> None:
        idx = self.selected_command_index()
        if idx < 0:
            return
        dlg = CommandDialog(self, self.commands[idx])
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.commands[idx] = dlg.get_data()
            self.refresh_command_table()
            self._save_all()

    def delete_command(self) -> None:
        idx = self.selected_command_index()
        if idx < 0:
            return
        del self.commands[idx]
        self.refresh_command_table()
        self._save_all()

    def send_selected_command(self) -> None:
        idx = self.selected_command_index()
        if idx < 0:
            QMessageBox.information(self, "情報", "送信する電文を選択してください。")
            return
        try:
            req = build_transaction(self.commands[idx])
        except Exception as ex:
            QMessageBox.warning(self, "電文エラー", str(ex))
            return
        self.pending_requests[req.request_id] = {"kind": "manual"}
        self.worker.submit_transaction(req)

    def add_sequence(self) -> None:
        dlg = SequenceDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if not data["name"]:
                QMessageBox.warning(self, "確認", "名前を入力してください。")
                return
            self.sequences.append(data)
            self.refresh_sequence_table()
            self._save_all()

    def edit_sequence(self) -> None:
        idx = self.selected_sequence_index()
        if idx < 0:
            return
        dlg = SequenceDialog(self, self.sequences[idx])
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.sequences[idx] = dlg.get_data()
            self.refresh_sequence_table()
            self._save_all()

    def delete_sequence(self) -> None:
        idx = self.selected_sequence_index()
        if idx < 0:
            return
        del self.sequences[idx]
        self.refresh_sequence_table()
        self._save_all()

    def run_selected_sequence(self) -> None:
        idx = self.selected_sequence_index()
        if idx < 0:
            QMessageBox.information(self, "情報", "実行するシーケンスを選択してください。")
            return
        if self.sequence_context:
            QMessageBox.information(self, "情報", "シーケンス実行中です。")
            return

        seq = self.sequences[idx]
        steps = parse_steps_text(seq.get("steps_text", ""))
        if not steps:
            QMessageBox.warning(self, "確認", "シーケンス手順がありません。")
            return

        self.sequence_context = {
            "name": seq.get("name", ""),
            "steps": steps,
            "loop_total": int(seq.get("loop_count", 1)),
            "loop_current": 1,
            "step_index": 0,
            "stopped": False,
        }
        self.append_log(f"シーケンス開始: {self.sequence_context['name']}")
        self._run_current_sequence_step()

    def stop_sequence(self) -> None:
        if self.sequence_context:
            self.sequence_context["stopped"] = True
            self.append_log("シーケンス停止要求を受け付けました。")

    def _run_current_sequence_step(self) -> None:
        if not self.sequence_context or self.sequence_context.get("stopped"):
            self.append_log("シーケンス停止")
            self.sequence_context = None
            return

        steps = self.sequence_context["steps"]
        idx = self.sequence_context["step_index"]

        if idx >= len(steps):
            if self.sequence_context["loop_current"] >= self.sequence_context["loop_total"]:
                self.append_log(f"シーケンス完了: {self.sequence_context['name']}")
                self.sequence_context = None
                return
            self.sequence_context["loop_current"] += 1
            self.sequence_context["step_index"] = 0
            idx = 0

        step = steps[idx]
        cmd = find_command(self.commands, step["command_name"])
        if not cmd:
            self.append_log(f"シーケンスエラー: コマンド未定義 {step['command_name']}")
            self.sequence_context = None
            return

        try:
            req = build_transaction(cmd)
        except Exception as ex:
            self.append_log(f"シーケンスエラー: {ex}")
            self.sequence_context = None
            return

        req.sequence_name = self.sequence_context["name"]
        req.step_index = idx
        req.retries_left = step["retries"]
        req.metadata = {"delay_ms": step["delay_ms"]}
        self.pending_requests[req.request_id] = {"kind": "sequence", "step": step}
        self.append_log(
            f"シーケンス実行: {req.sequence_name} loop={self.sequence_context['loop_current']}/{self.sequence_context['loop_total']} step={idx + 1}/{len(steps)} command={cmd['name']}"
        )
        self.worker.submit_transaction(req)

    def on_transaction_finished(self, result: dict) -> None:
        self.lbl_last_time.setText(f"通信時間: {result['elapsed_ms']} ms")
        self.append_log(
            f"結果[{result['name']}] ok={result['ok']} message={result['message']} time={result['elapsed_ms']}ms"
        )
        self.append_log(f"TX HEX   : {result.get('tx_hex', '')}")
        self.append_log(f"TX ASCII : {result.get('tx_ascii', '')}")
        self.append_log(f"RX HEX   : {result.get('rx_hex', '')}")
        self.append_log(f"RX ASCII : {result.get('rx_ascii', '')}")

        req_meta = self.pending_requests.pop(result["request_id"], None)
        if not req_meta or req_meta.get("kind") != "sequence" or not self.sequence_context:
            return

        step = req_meta["step"]
        if not result["ok"] and step["retries"] > 0:
            step["retries"] -= 1
            self.append_log(f"再送します。残り再送回数: {step['retries']}")
            QTimer.singleShot(int(step["delay_ms"]), self._run_current_sequence_step)
            return

        self.sequence_context["step_index"] += 1
        delay_ms = int(step["delay_ms"])
        QTimer.singleShot(delay_ms, self._run_current_sequence_step)

    def _save_all(self) -> None:
        self.config = {
            "serial_settings": self.collect_serial_settings(),
            "commands": self.commands,
            "sequences": self.sequences,
            "self_test": self.collect_self_test_settings(),
        }
        save_config(self.config, CONFIG_PATH)

    def export_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "設定を別名保存", str(CONFIG_PATH), "JSON (*.json)")
        if not path:
            return
        self._save_all()
        save_config(self.config, Path(path))
        self.append_log(f"設定を書き出しました: {path}")

    def import_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "設定を読み込む", str(CONFIG_PATH), "JSON (*.json)")
        if not path:
            return
        self.config = load_config(Path(path))
        self.commands = list(self.config.get("commands", []))
        self.sequences = list(self.config.get("sequences", []))
        self._load_config_into_ui()
        self.refresh_ports()
        self.refresh_command_table()
        self.refresh_sequence_table()
        self.append_log(f"設定を読み込みました: {path}")


def build_transaction(cmd: dict[str, Any]) -> TransactionRequest:
    payload = parse_payload(cmd.get("payload_type", "TEXT"), cmd.get("payload", ""), cmd.get("encoding", "ascii"))
    term_hex = normalize_hex(cmd.get("terminator_hex", ""))
    terminator = bytes.fromhex(term_hex) if term_hex else None
    request_id = f"REQ-{time.time_ns()}"
    return TransactionRequest(
        request_id=request_id,
        name=cmd.get("name", "CMD"),
        payload=payload,
        expect_reply=bool(cmd.get("expect_reply", True)),
        timeout_ms=int(cmd.get("timeout_ms", 300)),
        terminator=terminator,
    )


def parse_payload(payload_type: str, payload: str, encoding: str) -> bytes:
    if payload_type == "HEX":
        clean = normalize_hex(payload)
        if not clean:
            raise ValueError("HEX電文が空です。")
        return bytes.fromhex(clean)
    text = payload.encode("utf-8").decode("unicode_escape")
    return text.encode(encoding)



def normalize_hex(text: str) -> str:
    clean = text.replace(" ", "").replace("-", "").replace("0x", "").replace("0X", "")
    if clean and len(clean) % 2 != 0:
        raise ValueError("HEX文字列は偶数桁で入力してください。")
    return clean



def safe_ascii(data: bytes | bytearray) -> str:
    chars = []
    for b in data:
        chars.append(chr(b) if 0x20 <= b <= 0x7E else ".")
    return "".join(chars)



def bytes_to_hex(data: bytes | bytearray) -> str:
    return " ".join(f"{b:02X}" for b in data)



def parse_steps_text(text: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"シーケンス行の形式が不正です: {line}")
        steps.append(
            {
                "command_name": parts[0],
                "retries": int(parts[1]),
                "delay_ms": int(parts[2]),
            }
        )
    return steps



def find_command(commands: list[dict[str, Any]], name: str) -> Optional[dict[str, Any]]:
    for cmd in commands:
        if cmd.get("name") == name:
            return cmd
    return None



def map_bytesize(value: int) -> int:
    mapping = {
        5: serial.FIVEBITS,
        6: serial.SIXBITS,
        7: serial.SEVENBITS,
        8: serial.EIGHTBITS,
    }
    return mapping.get(int(value), serial.EIGHTBITS)



def map_parity(value: str) -> str:
    mapping = {
        "N": serial.PARITY_NONE,
        "E": serial.PARITY_EVEN,
        "O": serial.PARITY_ODD,
        "M": serial.PARITY_MARK,
        "S": serial.PARITY_SPACE,
    }
    return mapping.get(value, serial.PARITY_NONE)



def map_stopbits(value: float) -> float:
    if float(value) == 1.5:
        return serial.STOPBITS_ONE_POINT_FIVE
    if float(value) == 2:
        return serial.STOPBITS_TWO
    return serial.STOPBITS_ONE



def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        save_config(DEFAULT_CONFIG, path)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def save_config(config: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)



def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    pg.setConfigOptions(antialias=True)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
