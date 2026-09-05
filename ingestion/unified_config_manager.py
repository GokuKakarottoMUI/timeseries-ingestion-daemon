#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Config Manager
=======================
GUI quản lý toàn bộ config JSON của pipeline (PySide6 / Qt — lõi C++),
kèm nút chạy/dừng daemon fetch và xem log realtime.

Triết lý:
- Performance cao: Model/View của Qt cập nhật TỪNG cell, KHÔNG rebuild cả tree
  mỗi thao tác; search dùng QSortFilterProxyModel (filter realtime, không dựng lại
  model). Đọc JSON bằng orjson (lõi Rust). Ghi ATOMIC (file tạm + os.replace).
- Giữ NGUYÊN 100% logic nghiệp vụ & cấu trúc JSON của bản cũ — round-trip
  load→save không đổi/mất dữ liệu. Ghi giữ indent=4, ensure_ascii=False.

Quản lý 5 file (đều ở cùng thư mục config/):
  exchange_configs.json · historical_data_config.json · symbols_config.json
  all_timeframes.json · continuous_fetch_mode.json
"""

import os
import re
import sys
import json
import copy
import signal
import tempfile
from pathlib import Path

import orjson

from PySide6.QtCore import (Qt, QSortFilterProxyModel, QSettings, Signal,
                            QProcess, QProcessEnvironment)
from PySide6.QtGui import (QStandardItemModel, QStandardItem, QPalette, QColor,
                           QAction, QFont)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGridLayout, QLabel, QPushButton, QLineEdit, QCheckBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QTreeView, QTableView, QHeaderView,
    QAbstractItemView, QScrollArea, QGroupBox, QPlainTextEdit, QSplitter,
    QDialog, QDialogButtonBox, QMessageBox, QInputDialog, QFileDialog,
    QStatusBar, QStyleFactory, QStyledItemDelegate,
)

# ============================ Hằng số ============================
ROLE_META = Qt.UserRole + 1          # gắn metadata (node dict / path) vào item
TF_KEY_RE = re.compile(r'^\d+[mhdwM]$')

CONFIG_FILES = {
    'exchange':   'exchange_configs.json',
    'historical': 'historical_data_config.json',
    'symbols':    'symbols_config.json',
    'timeframes': 'all_timeframes.json',
    'continuous': 'continuous_fetch_mode.json',
}

GREEN = QColor('#1f6f43')
RED = QColor('#7a2230')
GREEN_FG = QColor('#7ee2a8')
RED_FG = QColor('#f1a7b0')


# ====================== Helper logic (port nguyên) ======================
def parse_timeframe_key(key: str):
    """Parse key '1h'/'30m'/'1d'/'1w'/'1M' → tổng số phút. None nếu sai."""
    m = re.match(r'^(\d+)([mhdwM])', key)
    if not m:
        return None
    value, unit = int(m.group(1)), m.group(2)
    if unit == 'm':
        return value
    if unit == 'h':
        return value * 60
    if unit == 'd':
        return value * 24 * 60
    if unit == 'w':
        return value * 7 * 24 * 60
    if unit == 'M':                       # tháng = 31 ngày (giữ như bản cũ)
        return value * 31 * 24 * 60
    return None


def minutes_to_other_units(minutes: int):
    """Chuyển minutes → seconds/hours (giữ logic bản cũ)."""
    seconds = minutes * 60
    hours = minutes // 60 if minutes >= 60 else None
    return {'seconds': seconds, 'hours': hours}


# ====================== Tầng IO: đọc orjson / ghi atomic ======================
def load_json(path: Path):
    with open(path, 'rb') as f:
        return orjson.loads(f.read())


def save_json_atomic(path: Path, data) -> None:
    """Ghi an toàn: serialize → file tạm cùng thư mục → os.replace (atomic POSIX).
    Giữ indent=4, ensure_ascii=False để khớp format file gốc (git diff sạch)."""
    text = json.dumps(data, indent=4, ensure_ascii=False)
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)            # atomic
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ====================== Dialog: Thêm/Sửa Timeframe ======================
class TimeframeDialog(QDialog):
    """Dialog thêm/sửa timeframe chuẩn (port logic timeframe_dialog cũ)."""

    def __init__(self, parent, timeframes_data, edit_key=None):
        super().__init__(parent)
        self.timeframes_data = timeframes_data
        self.edit_key = edit_key
        self.result_payload = None
        self.setWindowTitle("Thêm Timeframe" if not edit_key else f"Sửa Timeframe: {edit_key}")
        self.setMinimumWidth(380)

        form = QFormLayout(self)
        self.key_edit = QLineEdit()
        self.active_cb = QCheckBox("Active (Training/Features)")
        self.featured_cb = QCheckBox("Active Featured")
        self.prediction_cb = QCheckBox("Active Prediction (Output)")

        existing = (timeframes_data.get('timeframes', {}) or {}).get(edit_key) if edit_key else None
        if existing:
            self.key_edit.setText(edit_key)
            self.active_cb.setChecked(existing.get('active', False))
            self.featured_cb.setChecked(existing.get('active_featured', False))
            self.prediction_cb.setChecked(existing.get('active_prediction', False))

        form.addRow("Key (vd: 1h, 30m, 1d, 1w, 1M):", self.key_edit)
        form.addRow(self.active_cb)
        form.addRow(self.featured_cb)
        form.addRow(self.prediction_cb)
        info = QLabel("Các giá trị minutes/seconds/hours sẽ được tự động tính toán.")
        info.setStyleSheet("color:#5aa9e6;")
        form.addRow(info)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _on_accept(self):
        key = self.key_edit.text().strip()
        if not key:
            QMessageBox.critical(self, "Lỗi", "Vui lòng nhập key")
            return
        if not TF_KEY_RE.match(key):
            QMessageBox.critical(self, "Lỗi", "Key phải có định dạng như: 1m, 1h, 1d, 1w, 1M")
            return
        tfs = self.timeframes_data.setdefault('timeframes', {})
        if key in tfs and key != self.edit_key:
            QMessageBox.critical(self, "Lỗi", f"Key '{key}' đã tồn tại")
            return
        minutes = parse_timeframe_key(key)
        if minutes is None:
            QMessageBox.critical(self, "Lỗi", "Key không hợp lệ")
            return
        units = minutes_to_other_units(minutes)
        data = {
            'active': self.active_cb.isChecked(),
            'active_featured': self.featured_cb.isChecked(),
            'active_prediction': self.prediction_cb.isChecked(),
            'minutes': minutes,
            'seconds': units['seconds'],
        }
        if units['hours'] and minutes % 60 == 0:
            data['hours'] = units['hours']
        self.result_payload = (key, data)
        self.accept()


# ====================== Dialog: Thêm/Sửa Custom Timeframe ======================
class CustomTimeframeDialog(QDialog):
    """Dialog thêm/sửa custom timeframe (port logic custom_timeframe_dialog cũ)."""

    def __init__(self, parent, timeframes_data, edit_key=None):
        super().__init__(parent)
        self.timeframes_data = timeframes_data
        self.edit_key = edit_key
        self.result_payload = None
        self.setWindowTitle("Thêm Custom Timeframe" if not edit_key else f"Sửa Custom Timeframe: {edit_key}")
        self.setMinimumWidth(440)

        form = QFormLayout(self)
        self.key_edit = QLineEdit()
        self.active_cb = QCheckBox("Active (Training/Features)")
        self.featured_cb = QCheckBox("Active Featured")
        self.prediction_cb = QCheckBox("Active Prediction (Output)")
        self.source_combo = QComboBox()
        self.source_combo.setEditable(True)
        self.source_combo.addItems(list((timeframes_data.get('timeframes', {}) or {}).keys()))

        ct = timeframes_data.get('custom_timeframes', {}) or {}
        existing = (ct.get('custom_intervals', {}) or {}).get(edit_key) if edit_key else None
        if existing:
            self.key_edit.setText(edit_key)
            self.active_cb.setChecked(existing.get('active', False))
            self.featured_cb.setChecked(existing.get('active_featured', False))
            self.prediction_cb.setChecked(existing.get('active_prediction', False))
            self.source_combo.setCurrentText(existing.get('source', ''))

        form.addRow("Key (vd: 45m, 90m, 3h):", self.key_edit)
        form.addRow(self.active_cb)
        form.addRow(self.featured_cb)
        form.addRow(self.prediction_cb)
        form.addRow("Source:", self.source_combo)
        info = QLabel("Lưu ý: Custom timeframe phải chia hết cho source timeframe.\n"
                      "VD: 45m dùng source 15m, 90m dùng source 30m.")
        info.setWordWrap(True)
        info.setStyleSheet("color:#5aa9e6;")
        form.addRow(info)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _on_accept(self):
        key = self.key_edit.text().strip()
        source = self.source_combo.currentText().strip()
        if not key:
            QMessageBox.critical(self, "Lỗi", "Vui lòng nhập key")
            return
        if not source:
            QMessageBox.critical(self, "Lỗi", "Vui lòng chọn source")
            return
        if not TF_KEY_RE.match(key):
            QMessageBox.critical(self, "Lỗi", "Key phải có định dạng như: 45m, 90m, 3h")
            return
        ct = self.timeframes_data.setdefault('custom_timeframes', {'enable': False, 'custom_intervals': {}})
        intervals = ct.setdefault('custom_intervals', {})
        if key in intervals and key != self.edit_key:
            QMessageBox.critical(self, "Lỗi", f"Key '{key}' đã tồn tại")
            return
        custom_minutes = parse_timeframe_key(key)
        if custom_minutes is None:
            QMessageBox.critical(self, "Lỗi", "Key không hợp lệ")
            return
        tfs = self.timeframes_data.get('timeframes', {}) or {}
        if source not in tfs:
            QMessageBox.critical(self, "Lỗi", "Source không hợp lệ")
            return
        source_minutes = tfs[source]['minutes']
        if custom_minutes % source_minutes != 0:
            QMessageBox.critical(self, "Lỗi",
                                 f"Custom timeframe {key} ({custom_minutes}m) không chia hết cho "
                                 f"source {source} ({source_minutes}m)")
            return
        data = {
            'active': self.active_cb.isChecked(),
            'active_featured': self.featured_cb.isChecked(),
            'active_prediction': self.prediction_cb.isChecked(),
            'minutes': custom_minutes,
            'source': source,
        }
        if custom_minutes >= 60 and custom_minutes % 60 == 0:
            data['hours'] = custom_minutes // 60
        self.result_payload = (key, data)
        self.accept()


# ====================== Cửa sổ chính ======================
class UnifiedConfigManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pipeline Config Manager")
        self.resize(1280, 820)

        self.settings = QSettings("timeseries-ingestion-daemon", "ConfigManager")
        self.dark = self.settings.value("dark", True, type=bool)

        self._guard = False              # chặn signal write-back khi build model

        # Controller tiến trình continuous_fetch.py (xem _start_fetch). GUI chỉ điều
        # khiển — mọi semaphore/async/hardening nằm trong api_fetch (chạy ở process con).
        self.proc = None
        self._paused = False
        self._outbuf = b""

        self.base_path = self.find_config_path()

        # Dữ liệu (giữ tên field như bản cũ cho dễ đối chiếu)
        self.exchange_data = {}
        self.historical_data = {}
        self.symbols_data = {}
        self.timeframes_data = {}
        self.continuous_data = {}

        self._build_ui()
        self.apply_theme(self.dark)
        self.load_all_configs()

    # ---------- Tìm thư mục config (port logic find_config_path) ----------
    def find_config_path(self) -> Path:
        current_dir = Path(__file__).parent
        need = {'exchange_configs.json', 'historical_data_config.json', 'symbols_config.json'}
        for root, _dirs, files in os.walk(current_dir):
            if need.issubset(set(files)):
                return Path(root)
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa file config")
        return Path(folder) if folder else Path.cwd()

    # =================== UI khung ===================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 8)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        self.tabs.addTab(self._build_exchange_tab(), "Exchanges")
        self.tabs.addTab(self._build_historical_tab(), "Historical Data")
        self.tabs.addTab(self._build_symbols_tab(), "Symbols")
        self.tabs.addTab(self._build_timeframes_tab(), "Timeframes")
        self.tabs.addTab(self._build_custom_tab(), "Custom Timeframes")
        self.tabs.addTab(self._build_continuous_tab(), "Continuous Fetch")

        # Thanh nút dưới
        bottom = QHBoxLayout()
        self.theme_btn = QPushButton("Dark / Light")
        self.theme_btn.clicked.connect(self.toggle_theme)
        bottom.addWidget(self.theme_btn)
        bottom.addStretch(1)
        reload_btn = QPushButton("Tải lại")
        reload_btn.clicked.connect(self.load_all_configs)
        save_btn = QPushButton("Lưu tất cả")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self.save_all)
        bottom.addWidget(reload_btn)
        bottom.addWidget(save_btn)
        root.addLayout(bottom)

        self.setStatusBar(QStatusBar())
        self.status("Khởi tạo...")

    def status(self, msg: str):
        self.statusBar().showMessage(msg)

    # ---------------- Tab Exchanges ----------------
    def _build_exchange_tab(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        splitter = QSplitter(Qt.Horizontal)
        lay.addWidget(splitter)

        self.exchange_model = QStandardItemModel()
        self.exchange_model.setHorizontalHeaderLabels(["Exchange", "Active", "Rate Limit"])
        self.exchange_view = QTableView()
        self.exchange_view.setModel(self.exchange_model)
        self.exchange_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.exchange_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.exchange_view.verticalHeader().setVisible(False)
        self.exchange_view.horizontalHeader().setStretchLastSection(True)
        self.exchange_view.setAlternatingRowColors(True)
        self.exchange_view.setShowGrid(True)
        self.exchange_model.itemChanged.connect(self._on_exchange_item_changed)

        splitter.addWidget(self.exchange_view)

        # Panel chi tiết (read-only — không full editor theo yêu cầu)
        detail = QWidget()
        dl = QVBoxLayout(detail)
        dl.addWidget(QLabel("Chi tiết (chỉ xem):"))
        self.exchange_detail = QPlainTextEdit()
        self.exchange_detail.setReadOnly(True)
        dl.addWidget(self.exchange_detail, 1)
        splitter.addWidget(detail)
        splitter.setSizes([560, 640])

        self.exchange_view.clicked.connect(self._show_exchange_detail)
        return w

    def _show_exchange_detail(self, index):
        row = index.row()
        name_item = self.exchange_model.item(row, 0)
        if not name_item:
            return
        name = name_item.text()
        cfg = self.exchange_data.get('exchange_configs', {}).get(name, {})
        self.exchange_detail.setPlainText(json.dumps(cfg, indent=4, ensure_ascii=False))

    def _on_exchange_item_changed(self, item):
        if self._guard or item.column() != 1:
            return
        name_item = self.exchange_model.item(item.row(), 0)
        name = name_item.text()
        cfg = self.exchange_data.get('exchange_configs', {}).get(name)
        if cfg is None:
            return
        checked = item.checkState() == Qt.Checked
        cfg['active'] = checked
        self._guard = True
        self._paint_bool_item(item, checked)
        self._guard = False
        self.status(f"Đã {'bật' if checked else 'tắt'} exchange {name}")

    def refresh_exchange(self):
        self._guard = True
        self.exchange_model.removeRows(0, self.exchange_model.rowCount())
        for name, cfg in self.exchange_data.get('exchange_configs', {}).items():
            name_item = QStandardItem(name)
            name_item.setEditable(False)
            active_item = QStandardItem()
            active_item.setCheckable(True)
            active_item.setEditable(False)
            checked = bool(cfg.get('active', False))
            active_item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self._paint_bool_item(active_item, checked)
            rate_item = QStandardItem(str(cfg.get('rate_limit', 'N/A')))
            rate_item.setEditable(False)
            self.exchange_model.appendRow([name_item, active_item, rate_item])
        self.exchange_view.resizeColumnsToContents()
        self._guard = False

    # ---------------- Tab Historical Data ----------------
    def _build_historical_tab(self):
        w = QWidget()
        outer = QVBoxLayout(w)
        self.hist_scroll = QScrollArea()
        self.hist_scroll.setWidgetResizable(True)
        outer.addWidget(self.hist_scroll)
        self.hist_inner = QWidget()
        self.hist_form = QFormLayout(self.hist_inner)
        self.hist_scroll.setWidget(self.hist_inner)
        return w

    def refresh_historical(self):
        # Dựng lại form (chỉ khi load/reload — KHÔNG phải mỗi thao tác)
        while self.hist_form.rowCount():
            self.hist_form.removeRow(0)
        self._build_hist_widgets(self.historical_data, "")

    def _build_hist_widgets(self, data, path):
        for key, value in data.items():
            cur = f"{path}.{key}" if path else key
            if isinstance(value, dict) and 'value' in value and 'type' in value:
                self.hist_form.addRow(f"{key}:", self._make_value_widget(cur, value['value'], value['type']))
            elif isinstance(value, dict):
                header = QLabel(key)
                header.setStyleSheet("color:#5aa9e6; font-weight:bold; margin-top:6px;")
                self.hist_form.addRow(header)
                self._build_hist_widgets(value, cur)
            else:
                self.hist_form.addRow(f"{key}:", self._make_value_widget(cur, value, type(value).__name__))

    def _make_value_widget(self, path, value, vtype):
        if vtype == 'bool':
            cb = QCheckBox()
            cb.setChecked(bool(value))
            cb.toggled.connect(lambda checked, p=path: self._set_hist(p, bool(checked)))
            return cb
        if vtype == 'int':
            sp = QSpinBox()
            sp.setRange(-2_000_000_000, 2_000_000_000)
            sp.setValue(int(value))
            sp.valueChanged.connect(lambda v, p=path: self._set_hist(p, int(v)))
            return sp
        if vtype == 'float':
            sp = QDoubleSpinBox()
            sp.setRange(-1e12, 1e12)
            sp.setDecimals(8)
            sp.setValue(float(value))
            sp.valueChanged.connect(lambda v, p=path: self._set_hist(p, float(v)))
            return sp
        le = QLineEdit(str(value))
        le.editingFinished.connect(lambda w=le, p=path: self._set_hist(p, w.text()))
        return le

    def _set_hist(self, path, new_value):
        try:
            self._set_nested_value(self.historical_data, path.split('.'), new_value)
            self.status(f"Đã cập nhật: {path} = {new_value}")
        except Exception as e:
            self.status(f"Lỗi cập nhật {path}: {e}")

    @staticmethod
    def _set_nested_value(data, path, value):
        for key in path[:-1]:
            data = data[key]
        node = data[path[-1]]
        if isinstance(node, dict) and 'value' in node:
            node['value'] = value
        else:
            data[path[-1]] = value

    # ---------------- Tab Symbols ----------------
    def _build_symbols_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # Hàng nút thêm
        add_row = QHBoxLayout()
        for text, fn in [("Thêm Market", self.add_market),
                         ("Thêm Symbols Config", self.add_symbols_config),
                         ("Thêm Symbols", self.add_symbols),
                         ("Thêm Variant", self.add_variant)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            add_row.addWidget(b)
        add_row.addStretch(1)
        lay.addLayout(add_row)

        # Hàng sửa/xóa + tìm kiếm
        edit_row = QHBoxLayout()
        edit_btn = QPushButton("Sửa")
        edit_btn.clicked.connect(self.edit_symbol_item)
        del_btn = QPushButton("Xóa")
        del_btn.clicked.connect(self.delete_symbol_item)
        edit_row.addWidget(edit_btn)
        edit_row.addWidget(del_btn)
        edit_row.addStretch(1)
        edit_row.addWidget(QLabel("Tìm kiếm:"))
        self.symbol_search = QLineEdit()
        self.symbol_search.setFixedWidth(240)
        self.symbol_search.textChanged.connect(self._on_symbol_filter)
        edit_row.addWidget(self.symbol_search)
        lay.addLayout(edit_row)

        # Tree + proxy (search filter, KHÔNG rebuild model)
        self.symbols_model = QStandardItemModel()
        self.symbols_model.setHorizontalHeaderLabels(["Symbol / Variant", "Active"])
        self.symbols_proxy = QSortFilterProxyModel()
        self.symbols_proxy.setSourceModel(self.symbols_model)
        self.symbols_proxy.setRecursiveFilteringEnabled(True)
        self.symbols_proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.symbols_proxy.setFilterKeyColumn(0)

        self.symbols_view = QTreeView()
        self.symbols_view.setModel(self.symbols_proxy)
        self.symbols_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.symbols_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.symbols_view.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.symbols_view.setAlternatingRowColors(True)
        self.symbols_view.setUniformRowHeights(True)
        self.symbols_model.itemChanged.connect(self._on_symbol_item_changed)
        lay.addWidget(self.symbols_view, 1)
        return w

    def _on_symbol_filter(self, text):
        self.symbols_proxy.setFilterFixedString(text)
        if text:
            self.symbols_view.expandAll()

    def refresh_symbols(self):
        self._guard = True
        self.symbols_model.removeRows(0, self.symbols_model.rowCount())
        root = self.symbols_model.invisibleRootItem()
        for m_name, m_data in self.symbols_data.get('market', {}).items():
            m_row = self._make_symbol_row(m_name, m_data, 1, [m_name])
            root.appendRow(m_row)
            for c_name, c_data in m_data.get('symbols_config', {}).items():
                c_row = self._make_symbol_row(c_name, c_data, 2, [m_name, c_name])
                m_row[0].appendRow(c_row)
                for s_name, s_data in c_data.get('symbols', {}).items():
                    s_row = self._make_symbol_row(s_name, s_data, 3, [m_name, c_name, s_name])
                    c_row[0].appendRow(s_row)
                    for v_name, v_data in s_data.get('variants', {}).items():
                        v_row = self._make_symbol_row(v_name, v_data, 4,
                                                      [m_name, c_name, s_name, v_name])
                        s_row[0].appendRow(v_row)
        self.symbols_view.expandAll()
        self._guard = False

    def _make_symbol_row(self, name, node, level, keys):
        # LƯU Ý: Qt copy Python dict khi setData → KHÔNG lưu ref node/parent.
        # Chỉ lưu đường dẫn keys (list string), điều hướng data SỐNG khi cần.
        meta = {'keys': list(keys), 'level': level}
        # market mặc định active=True nếu thiếu (giữ logic cũ), còn lại default False
        default_active = True if level == 1 else False
        checked = bool(node.get('active', default_active))
        name_item = QStandardItem(name)
        name_item.setEditable(False)
        name_item.setData(meta, ROLE_META)
        self._paint_symbol_name(name_item, checked)      # tô NGUYÊN dòng: xanh=true / đỏ=false
        active_item = QStandardItem()
        active_item.setCheckable(True)
        active_item.setEditable(False)
        active_item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        active_item.setData(meta, ROLE_META)
        self._paint_bool_item(active_item, checked)
        return [name_item, active_item]

    def _paint_symbol_name(self, name_item, checked):
        # tô nền xanh/đỏ cho cột TÊN (giữ nguyên text = tên symbol)
        name_item.setBackground(GREEN if checked else RED)
        name_item.setForeground(GREEN_FG if checked else RED_FG)

    def _resolve_symbol(self, keys):
        """Điều hướng TRỰC TIẾP trên data sống → (node, parent_container, key)."""
        market = self.symbols_data['market']
        parent = market
        node = market[keys[0]]
        if len(keys) >= 2:
            parent = node['symbols_config']; node = parent[keys[1]]
        if len(keys) >= 3:
            parent = node['symbols']; node = parent[keys[2]]
        if len(keys) >= 4:
            parent = node['variants']; node = parent[keys[3]]
        return node, parent, keys[-1]

    def _on_symbol_item_changed(self, item):
        if self._guard or item.column() != 1:
            return
        meta = item.data(ROLE_META)
        if not meta:
            return
        node, _parent, key = self._resolve_symbol(meta['keys'])
        checked = item.checkState() == Qt.Checked
        node['active'] = checked
        self._guard = True
        self._paint_bool_item(item, checked)
        name_item = self.symbols_model.itemFromIndex(item.index().siblingAtColumn(0))
        if name_item is not None:
            self._paint_symbol_name(name_item, checked)  # đồng bộ màu cột tên cùng dòng
        self._guard = False
        self.status(f"Đã {'bật' if checked else 'tắt'} {key}")

    def _selected_symbol_meta(self):
        idx = self.symbols_view.currentIndex()
        if not idx.isValid():
            return None
        src = self.symbols_proxy.mapToSource(idx)
        item = self.symbols_model.itemFromIndex(src.siblingAtColumn(0))
        return item.data(ROLE_META) if item else None

    def add_market(self):
        name, ok = QInputDialog.getText(self, "Thêm Market", "Nhập tên market:")
        if not ok or not name:
            return
        name = name.title()
        market = self.symbols_data.setdefault('market', {})
        if name in market:
            QMessageBox.warning(self, "Cảnh báo", f"Market '{name}' đã tồn tại!")
            return
        market[name] = {'active': True, 'symbols_config': {}}
        self.refresh_symbols()
        self.status(f"Đã thêm market: {name}")

    def add_symbols_config(self):
        meta = self._selected_symbol_meta()
        if not meta or meta['level'] != 1:
            QMessageBox.warning(self, "Yêu cầu chọn", "Vui lòng chọn Market để thêm Symbols Config!")
            return
        name, ok = QInputDialog.getText(self, "Thêm Symbols Config", "Nhập tên (VD: BTC, ETH):")
        if not ok or not name:
            return
        name = name.upper()
        node, _p, _k = self._resolve_symbol(meta['keys'])
        container = node['symbols_config']
        if name in container:
            QMessageBox.warning(self, "Cảnh báo", f"Symbols Config '{name}' đã tồn tại!")
            return
        container[name] = {'active': False, 'symbols': {}}
        self.refresh_symbols()
        self.status(f"Đã thêm symbols config: {name}")

    def add_symbols(self):
        meta = self._selected_symbol_meta()
        if not meta or meta['level'] != 2:
            QMessageBox.warning(self, "Yêu cầu chọn", "Vui lòng chọn Symbols Config để thêm Symbols!")
            return
        name, ok = QInputDialog.getText(self, "Thêm Symbols", "Nhập tên symbol (VD: BTCUSD):")
        if not ok or not name:
            return
        name = name.upper()
        node, _p, _k = self._resolve_symbol(meta['keys'])
        container = node['symbols']
        if name in container:
            QMessageBox.warning(self, "Cảnh báo", f"Symbol '{name}' đã tồn tại!")
            return
        container[name] = {'active': True, 'variants': {}}
        self.refresh_symbols()
        self.status(f"Đã thêm symbol: {name}")

    def add_variant(self):
        meta = self._selected_symbol_meta()
        if not meta or meta['level'] != 3:
            QMessageBox.warning(self, "Yêu cầu chọn", "Vui lòng chọn Symbol để thêm Variant!")
            return
        name, ok = QInputDialog.getText(self, "Thêm Variant", "Nhập tên variant:")
        if not ok or not name:
            return
        node, _p, _k = self._resolve_symbol(meta['keys'])
        container = node['variants']
        if name in container:
            QMessageBox.warning(self, "Cảnh báo", f"Variant '{name}' đã tồn tại!")
            return
        container[name] = {'active': True}
        self.refresh_symbols()
        self.status(f"Đã thêm variant: {name}")

    def edit_symbol_item(self):
        meta = self._selected_symbol_meta()
        if not meta:
            self.status("Vui lòng chọn item để sửa")
            return
        _node, parent, old = self._resolve_symbol(meta['keys'])
        new, ok = QInputDialog.getText(self, "Sửa tên", "Nhập tên mới:", text=old)
        if not ok or not new or new == old:
            return
        new = new.upper() if meta['level'] >= 2 else new.title()
        if new in parent:
            QMessageBox.warning(self, "Cảnh báo", f"Tên '{new}' đã tồn tại!")
            return
        # đổi key GIỮ NGUYÊN thứ tự: dựng dict mới rồi thay nội dung tại chỗ
        rebuilt = {(new if k == old else k): v for k, v in parent.items()}
        parent.clear()
        parent.update(rebuilt)
        self.refresh_symbols()
        self.status(f"Đã đổi tên: {old} → {new}")

    def delete_symbol_item(self):
        meta = self._selected_symbol_meta()
        if not meta:
            self.status("Vui lòng chọn item để xóa")
            return
        _node, parent, key = self._resolve_symbol(meta['keys'])
        if QMessageBox.question(self, "Xác nhận xóa", f"Bạn có chắc muốn xóa '{key}'?") \
                != QMessageBox.Yes:
            return
        parent.pop(key, None)
        self.refresh_symbols()
        self.status(f"Đã xóa: {key}")

    # ---------------- Tab Timeframes ----------------
    def _build_timeframes_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        ctrl = QHBoxLayout()
        for text, fn in [("Thêm", self.add_timeframe), ("Sửa", self.edit_timeframe),
                         ("Xóa", self.delete_timeframe)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            ctrl.addWidget(b)
        ctrl.addStretch(1)
        lay.addLayout(ctrl)

        self.tf_model = QStandardItemModel()
        self.tf_model.setHorizontalHeaderLabels(
            ["Key", "Active", "Featured", "Prediction", "Minutes", "Seconds", "Hours"])
        self.tf_view = QTableView()
        self.tf_view.setModel(self.tf_model)
        self.tf_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tf_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tf_view.verticalHeader().setVisible(False)
        self.tf_view.horizontalHeader().setStretchLastSection(True)
        self.tf_view.setAlternatingRowColors(True)
        self.tf_view.setShowGrid(True)
        self.tf_view.doubleClicked.connect(self.edit_timeframe)
        self.tf_model.itemChanged.connect(self._on_tf_item_changed)
        lay.addWidget(self.tf_view, 1)
        return w

    def refresh_timeframes(self):
        self._guard = True
        self.tf_model.removeRows(0, self.tf_model.rowCount())
        items = sorted((self.timeframes_data.get('timeframes', {}) or {}).items(),
                       key=lambda x: x[1].get('minutes', 0))
        for key, val in items:
            self.tf_model.appendRow(self._tf_row(key, val, 'timeframes'))
        self.tf_view.resizeColumnsToContents()
        self._guard = False

    def _tf_row(self, key, val, kind, extra_col='seconds'):
        meta = {'key': key, 'kind': kind}
        key_item = QStandardItem(key)
        key_item.setEditable(False)
        key_item.setData(meta, ROLE_META)
        row = [key_item]
        for field in ('active', 'active_featured', 'active_prediction'):
            it = QStandardItem()
            it.setCheckable(True)
            it.setEditable(False)
            checked = bool(val.get(field, False))
            it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            it.setData(meta, ROLE_META)
            self._paint_bool_item(it, checked)
            row.append(it)
        minutes = QStandardItem(str(val.get('minutes', 0)))
        minutes.setEditable(False)
        extra = QStandardItem(str(val.get(extra_col, '')))
        extra.setEditable(False)
        hours = QStandardItem(str(val.get('hours', '')))
        hours.setEditable(False)
        row += [minutes, extra, hours]
        return row

    def _on_tf_item_changed(self, item):
        if self._guard or item.column() not in (1, 2, 3):
            return
        meta = item.data(ROLE_META)
        if not meta:
            return
        field = {1: 'active', 2: 'active_featured', 3: 'active_prediction'}[item.column()]
        checked = item.checkState() == Qt.Checked
        store = (self.timeframes_data['custom_timeframes']['custom_intervals']
                 if meta['kind'] == 'custom' else self.timeframes_data['timeframes'])
        if meta['key'] in store:
            store[meta['key']][field] = checked
            self._guard = True
            self._paint_bool_item(item, checked)
            self._guard = False
            self.status(f"Đã {'bật' if checked else 'tắt'} {field} cho '{meta['key']}'")

    def add_timeframe(self):
        dlg = TimeframeDialog(self, self.timeframes_data)
        if dlg.exec() == QDialog.Accepted and dlg.result_payload:
            key, data = dlg.result_payload
            tfs = self.timeframes_data.setdefault('timeframes', {})
            tfs[key] = data
            self.timeframes_data['timeframes'] = dict(
                sorted(tfs.items(), key=lambda x: x[1].get('minutes', 0)))
            self.refresh_timeframes()
            self.status(f"Đã thêm timeframe '{key}'")

    def edit_timeframe(self, *args):
        idx = self.tf_view.currentIndex()
        if not idx.isValid():
            self.status("Vui lòng chọn một timeframe để sửa")
            return
        key = self.tf_model.item(idx.row(), 0).text()
        dlg = TimeframeDialog(self, self.timeframes_data, edit_key=key)
        if dlg.exec() == QDialog.Accepted and dlg.result_payload:
            new_key, data = dlg.result_payload
            tfs = self.timeframes_data['timeframes']
            if key in tfs:
                del tfs[key]
            tfs[new_key] = data
            self.timeframes_data['timeframes'] = dict(
                sorted(tfs.items(), key=lambda x: x[1].get('minutes', 0)))
            self.refresh_timeframes()
            self.status(f"Đã cập nhật timeframe '{new_key}'")

    def delete_timeframe(self):
        idx = self.tf_view.currentIndex()
        if not idx.isValid():
            self.status("Vui lòng chọn một timeframe để xóa")
            return
        key = self.tf_model.item(idx.row(), 0).text()
        if QMessageBox.question(self, "Xác nhận", f"Xóa timeframe '{key}'?") != QMessageBox.Yes:
            return
        self.timeframes_data.get('timeframes', {}).pop(key, None)
        self.refresh_timeframes()
        self.status(f"Đã xóa timeframe '{key}'")

    # ---------------- Tab Custom Timeframes ----------------
    def _build_custom_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.custom_enable = QCheckBox("Bật Custom Timeframes")
        self.custom_enable.toggled.connect(self._on_custom_enable)
        lay.addWidget(self.custom_enable)

        ctrl = QHBoxLayout()
        for text, fn in [("Thêm", self.add_custom_timeframe), ("Sửa", self.edit_custom_timeframe),
                         ("Xóa", self.delete_custom_timeframe)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            ctrl.addWidget(b)
        ctrl.addStretch(1)
        lay.addLayout(ctrl)

        self.custom_model = QStandardItemModel()
        self.custom_model.setHorizontalHeaderLabels(
            ["Key", "Active", "Featured", "Prediction", "Minutes", "Source", "Hours"])
        self.custom_view = QTableView()
        self.custom_view.setModel(self.custom_model)
        self.custom_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.custom_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.custom_view.verticalHeader().setVisible(False)
        self.custom_view.horizontalHeader().setStretchLastSection(True)
        self.custom_view.setAlternatingRowColors(True)
        self.custom_view.setShowGrid(True)
        self.custom_view.doubleClicked.connect(self.edit_custom_timeframe)
        self.custom_model.itemChanged.connect(self._on_tf_item_changed)
        lay.addWidget(self.custom_view, 1)
        return w

    def _on_custom_enable(self, checked):
        if self._guard:
            return
        ct = self.timeframes_data.setdefault('custom_timeframes', {'enable': False, 'custom_intervals': {}})
        ct['enable'] = bool(checked)
        self.status("Đã cập nhật trạng thái Custom Timeframes")

    def refresh_custom(self):
        self._guard = True
        ct = self.timeframes_data.get('custom_timeframes', {}) or {}
        self.custom_enable.setChecked(bool(ct.get('enable', False)))
        self.custom_model.removeRows(0, self.custom_model.rowCount())
        items = sorted((ct.get('custom_intervals', {}) or {}).items(),
                       key=lambda x: x[1].get('minutes', 0))
        for key, val in items:
            self.custom_model.appendRow(self._tf_row(key, val, 'custom', extra_col='source'))
        self.custom_view.resizeColumnsToContents()
        self._guard = False

    def add_custom_timeframe(self):
        dlg = CustomTimeframeDialog(self, self.timeframes_data)
        if dlg.exec() == QDialog.Accepted and dlg.result_payload:
            key, data = dlg.result_payload
            ct = self.timeframes_data.setdefault('custom_timeframes',
                                                 {'enable': False, 'custom_intervals': {}})
            intervals = ct.setdefault('custom_intervals', {})
            intervals[key] = data
            ct['custom_intervals'] = dict(sorted(intervals.items(),
                                                 key=lambda x: x[1].get('minutes', 0)))
            self.refresh_custom()
            self.status(f"Đã thêm custom timeframe '{key}'")

    def edit_custom_timeframe(self, *args):
        idx = self.custom_view.currentIndex()
        if not idx.isValid():
            self.status("Vui lòng chọn một custom timeframe để sửa")
            return
        key = self.custom_model.item(idx.row(), 0).text()
        dlg = CustomTimeframeDialog(self, self.timeframes_data, edit_key=key)
        if dlg.exec() == QDialog.Accepted and dlg.result_payload:
            new_key, data = dlg.result_payload
            intervals = self.timeframes_data['custom_timeframes']['custom_intervals']
            if key in intervals:
                del intervals[key]
            intervals[new_key] = data
            self.timeframes_data['custom_timeframes']['custom_intervals'] = dict(
                sorted(intervals.items(), key=lambda x: x[1].get('minutes', 0)))
            self.refresh_custom()
            self.status(f"Đã cập nhật custom timeframe '{new_key}'")

    def delete_custom_timeframe(self):
        idx = self.custom_view.currentIndex()
        if not idx.isValid():
            self.status("Vui lòng chọn một custom timeframe để xóa")
            return
        key = self.custom_model.item(idx.row(), 0).text()
        if QMessageBox.question(self, "Xác nhận", f"Xóa custom timeframe '{key}'?") != QMessageBox.Yes:
            return
        self.timeframes_data.get('custom_timeframes', {}).get('custom_intervals', {}).pop(key, None)
        self.refresh_custom()
        self.status(f"Đã xóa custom timeframe '{key}'")

    # ---------------- Tab Continuous Fetch (MỚI) ----------------
    def _build_continuous_tab(self):
        w = QWidget()
        outer = QVBoxLayout(w)

        # --- Nhóm 1: Cấu hình (giữ logic cũ) ---
        box = QGroupBox("Cấu hình (continuous_fetch_mode.json)")
        form = QFormLayout(box)
        self.cont_continuous = QCheckBox("Bật chế độ continuous")
        self.cont_continuous.toggled.connect(
            lambda v: self._set_continuous('continuous', bool(v)))
        form.addRow("continuous:", self.cont_continuous)
        self.cont_fetch_interval = self._cont_spin('fetch_interval')
        form.addRow("fetch_interval (s):", self.cont_fetch_interval)
        self.cont_sleep_interval = self._cont_spin('sleep_interval')
        form.addRow("sleep_interval (s):", self.cont_sleep_interval)
        self.cont_continuous_sleep = self._cont_spin('continuous_sleep_interval')
        form.addRow("continuous_sleep_interval (s):", self.cont_continuous_sleep)
        outer.addWidget(box)

        # --- Nhóm 2: Điều khiển tiến trình continuous_fetch.py ---
        ctrl = QGroupBox("Điều khiển tiến trình (continuous_fetch.py)")
        cl = QVBoxLayout(ctrl)
        row = QHBoxLayout()
        self.btn_start = QPushButton("▶ Bắt đầu")
        self.btn_start.setObjectName("primary")
        self.btn_pause = QPushButton("Tạm dừng")
        self.btn_resume = QPushButton("Tiếp tục")
        self.btn_stop = QPushButton("Thoát")
        self.btn_start.clicked.connect(self._start_fetch)
        self.btn_pause.clicked.connect(self._pause_fetch)
        self.btn_resume.clicked.connect(self._resume_fetch)
        self.btn_stop.clicked.connect(self._stop_fetch)
        for b in (self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop):
            row.addWidget(b)
        row.addStretch(1)
        self.proc_status = QLabel("● Đã dừng")
        row.addWidget(self.proc_status)
        cl.addLayout(row)
        outer.addWidget(ctrl)

        # --- Nhóm 3: Log realtime ---
        log_box = QGroupBox("Log")
        ll = QVBoxLayout(log_box)
        top = QHBoxLayout()
        self.autoscroll_cb = QCheckBox("Tự cuộn")
        self.autoscroll_cb.setChecked(True)
        clear_btn = QPushButton("Xóa log")
        clear_btn.clicked.connect(lambda: self.log_view.clear())
        top.addWidget(self.autoscroll_cb)
        top.addStretch(1)
        top.addWidget(clear_btn)
        ll.addLayout(top)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)     # ring buffer → RAM bounded
        mono = QFont("Monospace"); mono.setStyleHint(QFont.TypeWriter); mono.setPointSize(10)
        self.log_view.setFont(mono)
        ll.addWidget(self.log_view, 1)
        outer.addWidget(log_box, 1)

        self._update_proc_buttons()
        return w

    # ---------------- Điều khiển tiến trình continuous_fetch ----------------
    def _fetch_running(self):
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def _append_log(self, text):
        if text == "":
            return
        self.log_view.appendPlainText(text)
        if self.autoscroll_cb.isChecked():
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _start_fetch(self):
        if self._fetch_running():
            return                                   # semaphore = 1: không chạy chồng
        errs = self.validate_before_save()
        if errs:
            QMessageBox.critical(self, "Không thể chạy — lỗi validate",
                                 "\n".join(f"• {e}" for e in errs))
            return
        self.save_all()                              # tiến trình con đọc config TỪ ĐĨA
        # Chạy bằng `-m` với cwd = repo root: daemon dùng absolute import theo package,
        # nên cwd phải là thư mục CHA của `ingestion/`, không phải chính nó.
        project_root = Path(__file__).resolve().parent.parent
        module = "ingestion.continuous_fetch"
        self._outbuf = b""
        self._paused = False
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")          # log realtime, không block-buffer
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(project_root))
        self.proc.readyReadStandardOutput.connect(self._read_proc)
        self.proc.finished.connect(self._on_proc_finished)
        self.proc.errorOccurred.connect(self._on_proc_error)
        self._append_log(f"$ {sys.executable} -m {module}")
        self.proc.start(sys.executable, ["-m", module])
        self._update_proc_buttons()
        self.status("Đã khởi chạy continuous_fetch")

    def _read_proc(self):
        if self.proc is None:
            return
        self._outbuf += bytes(self.proc.readAllStandardOutput())
        if b"\n" not in self._outbuf:
            return
        *lines, self._outbuf = self._outbuf.split(b"\n")  # giữ lại đuôi dở
        self._append_log(b"\n".join(lines).decode("utf-8", "replace"))

    def _send_signal(self, sig, note):
        if not self._fetch_running():
            return
        pid = self.proc.processId()
        if pid <= 0:                                 # GUARD: chặn os.kill(0/-1) = giết group
            return
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, OSError) as e:   # tiến trình có thể vừa chết
            self._append_log(f"[GUI] Không gửi được tín hiệu: {e}")
            return
        self.status(note)

    def _pause_fetch(self):
        self._send_signal(signal.SIGUSR1, "Đã gửi tín hiệu TẠM DỪNG")
        self._paused = True
        self._update_proc_buttons()

    def _resume_fetch(self):
        self._send_signal(signal.SIGUSR2, "Đã gửi tín hiệu TIẾP TỤC")
        self._paused = False
        self._update_proc_buttons()

    def _stop_fetch(self):
        self._send_signal(signal.SIGTERM, "Đã gửi tín hiệu THOÁT")

    def _on_proc_finished(self, code, status):
        if self._outbuf:                             # flush đuôi còn lại
            self._append_log(self._outbuf.decode("utf-8", "replace"))
            self._outbuf = b""
        self._append_log(f"[GUI] Tiến trình kết thúc (exit code {code}).")
        self.proc = None
        self._paused = False
        self._update_proc_buttons()

    def _on_proc_error(self, err):
        self._append_log(f"[GUI] Lỗi tiến trình: {err}")
        if not self._fetch_running():
            self.proc = None
            self._paused = False
            self._update_proc_buttons()

    def _update_proc_buttons(self):
        running = self._fetch_running()
        self.btn_start.setEnabled(not running)
        self.btn_pause.setEnabled(running and not self._paused)
        self.btn_resume.setEnabled(running and self._paused)
        self.btn_stop.setEnabled(running)
        if not running:
            self.proc_status.setText("● Đã dừng")
            self.proc_status.setStyleSheet("color:#a0a4ab;")
        elif self._paused:
            self.proc_status.setText(f"● Đã tạm dừng (PID {self.proc.processId()})")
            self.proc_status.setStyleSheet("color:#e0b050;")
        else:
            self.proc_status.setText(f"● Đang chạy (PID {self.proc.processId()})")
            self.proc_status.setStyleSheet("color:#7ee2a8;")

    def closeEvent(self, event):
        # Đóng app khi đang chạy → hỏi rồi dừng sạch (SIGTERM → SIGKILL fallback).
        if self._fetch_running():
            r = QMessageBox.question(
                self, "Đang chạy continuous_fetch",
                "Tiến trình fetch đang chạy. Dừng tiến trình và thoát?",
                QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                event.ignore()
                return
            pid = self.proc.processId()
            if pid > 0:
                try:
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
            if self.proc is not None and not self.proc.waitForFinished(3000):
                self.proc.kill()                     # SIGKILL fallback — chống orphan
                self.proc.waitForFinished(2000)
        event.accept()

    def _cont_spin(self, name):
        sp = QSpinBox()
        sp.setRange(0, 2_000_000_000)
        sp.valueChanged.connect(lambda v, n=name: self._set_continuous(n, int(v)))
        return sp

    def _set_continuous(self, name, value):
        if self._guard:
            return
        fm = self.continuous_data.setdefault('fetch_mode', {})
        node = fm.setdefault(name, {'name': name, 'value': value})
        node['value'] = value
        self.status(f"Đã cập nhật continuous.{name} = {value}")

    def refresh_continuous(self):
        self._guard = True
        fm = self.continuous_data.get('fetch_mode', {}) or {}
        self.cont_continuous.setChecked(bool(fm.get('continuous', {}).get('value', False)))
        self.cont_fetch_interval.setValue(int(fm.get('fetch_interval', {}).get('value', 0)))
        self.cont_sleep_interval.setValue(int(fm.get('sleep_interval', {}).get('value', 0)))
        self.cont_continuous_sleep.setValue(int(fm.get('continuous_sleep_interval', {}).get('value', 0)))
        self._guard = False

    # =================== Load / Save ===================
    def load_all_configs(self):
        try:
            self.exchange_data = load_json(self.base_path / CONFIG_FILES['exchange'])
            self.historical_data = load_json(self.base_path / CONFIG_FILES['historical'])
            self.symbols_data = load_json(self.base_path / CONFIG_FILES['symbols'])
            self.timeframes_data = load_json(self.base_path / CONFIG_FILES['timeframes'])
            self.continuous_data = load_json(self.base_path / CONFIG_FILES['continuous'])
        except Exception as e:
            QMessageBox.critical(self, "Lỗi tải config", str(e))
            self.status(f"Không thể tải config: {e}")
            return
        self.refresh_exchange()
        self.refresh_historical()
        self.refresh_symbols()
        self.refresh_timeframes()
        self.refresh_custom()
        self.refresh_continuous()
        self.status("Đã tải tất cả config thành công!")

    def validate_before_save(self):
        """Validate nhẹ (defensive) trước khi ghi. Trả về list lỗi."""
        errors = []
        tfs = self.timeframes_data.get('timeframes', {}) or {}
        for key, val in tfs.items():
            if not TF_KEY_RE.match(key):
                errors.append(f"Timeframe key sai định dạng: '{key}'")
            if 'minutes' not in val:
                errors.append(f"Timeframe '{key}' thiếu 'minutes'")
        ct = self.timeframes_data.get('custom_timeframes', {}) or {}
        for key, val in (ct.get('custom_intervals', {}) or {}).items():
            if not TF_KEY_RE.match(key):
                errors.append(f"Custom TF key sai định dạng: '{key}'")
            src = val.get('source')
            if src not in tfs:
                errors.append(f"Custom TF '{key}' có source '{src}' không tồn tại")
            elif val.get('minutes', 0) % tfs[src]['minutes'] != 0:
                errors.append(f"Custom TF '{key}' không chia hết cho source '{src}'")
        return errors

    def save_all(self):
        errors = self.validate_before_save()
        if errors:
            QMessageBox.critical(self, "Không thể lưu — có lỗi validate",
                                 "\n".join(f"• {e}" for e in errors))
            self.status("Lưu bị chặn do lỗi validate.")
            return
        targets = [
            (CONFIG_FILES['exchange'], self.exchange_data),
            (CONFIG_FILES['historical'], self.historical_data),
            (CONFIG_FILES['symbols'], self.symbols_data),
            (CONFIG_FILES['timeframes'], self.timeframes_data),
            (CONFIG_FILES['continuous'], self.continuous_data),
        ]
        failed = []
        for fname, data in targets:
            try:
                save_json_atomic(self.base_path / fname, data)
            except Exception as e:
                failed.append(f"{fname}: {e}")
        if failed:
            QMessageBox.critical(self, "Lỗi khi lưu", "\n".join(failed))
            self.status("Có file lưu thất bại.")
        else:
            self.status("Đã lưu tất cả config thành công (atomic)!")

    # =================== Theme ===================
    def _paint_bool_item(self, item, checked):
        item.setBackground(GREEN if checked else RED)
        item.setForeground(GREEN_FG if checked else RED_FG)
        item.setText("  True" if checked else "  False")
        item.setTextAlignment(Qt.AlignCenter)

    def toggle_theme(self):
        self.dark = not self.dark
        self.settings.setValue("dark", self.dark)
        self.apply_theme(self.dark)

    def apply_theme(self, dark: bool):
        app = QApplication.instance()
        app.setStyle(QStyleFactory.create("Fusion"))
        pal = QPalette()
        if dark:
            base = QColor(30, 31, 34)
            alt = QColor(38, 40, 44)
            text = QColor(220, 223, 228)
            accent = QColor(64, 132, 214)
            pal.setColor(QPalette.Window, base)
            pal.setColor(QPalette.WindowText, text)
            pal.setColor(QPalette.Base, QColor(24, 25, 28))
            pal.setColor(QPalette.AlternateBase, alt)
            pal.setColor(QPalette.Text, text)
            pal.setColor(QPalette.Button, alt)
            pal.setColor(QPalette.ButtonText, text)
            pal.setColor(QPalette.Highlight, accent)
            pal.setColor(QPalette.HighlightedText, Qt.white)
            pal.setColor(QPalette.ToolTipBase, base)
            pal.setColor(QPalette.ToolTipText, text)
            qss = """
            QWidget { font-size: 13px; }
            QTabWidget::pane { border: 1px solid #3a3d42; border-radius: 6px; }
            QTabBar::tab { background:#2b2d31; padding:8px 16px; margin-right:2px;
                           border-top-left-radius:6px; border-top-right-radius:6px; }
            QTabBar::tab:selected { background:#4084d6; color:white; }
            QPushButton { background:#34373d; border:1px solid #45484f; border-radius:6px;
                          padding:6px 14px; }
            QPushButton:hover { background:#3e424a; }
            QPushButton#primary { background:#2e7d46; border:1px solid #2e7d46; font-weight:bold; }
            QPushButton#primary:hover { background:#359152; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
                background:#1d1e21; border:1px solid #45484f; border-radius:5px; padding:4px; }
            QHeaderView::section { background:#2b2d31; padding:6px;
                                   border:none; border-right:1px solid #45484f;
                                   border-bottom:1px solid #45484f; }
            QTableView { gridline-color:#4a4e55; alternate-background-color:#23252a;
                         selection-background-color:#345a8a; }
            QTreeView { alternate-background-color:#23252a;
                        selection-background-color:#345a8a; }
            QTreeView::item { border-right:1px solid #3a3d42;
                              border-bottom:1px solid #2c2f34; padding:2px 0; }
            """
        else:
            pal = app.style().standardPalette()
            qss = """
            QPushButton#primary { background:#2e7d46; color:white; font-weight:bold;
                                  border-radius:6px; padding:6px 14px; }
            QTabBar::tab:selected { color:#1565c0; }
            QHeaderView::section { border-right:1px solid #c0c4cc;
                                   border-bottom:1px solid #c0c4cc; padding:6px; }
            QTableView { gridline-color:#c0c4cc; alternate-background-color:#f2f4f7; }
            QTreeView { alternate-background-color:#f2f4f7; }
            QTreeView::item { border-right:1px solid #dde0e5;
                              border-bottom:1px solid #eceef1; padding:2px 0; }
            """
        app.setPalette(pal)
        app.setStyleSheet(qss)
        # vẽ lại màu bool cho khớp (foreground phụ thuộc theme)
        if hasattr(self, 'exchange_model'):
            self.refresh_exchange()
            self.refresh_symbols()
            self.refresh_timeframes()
            self.refresh_custom()


def main():
    app = QApplication(sys.argv)
    win = UnifiedConfigManager()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
