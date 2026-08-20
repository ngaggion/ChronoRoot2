"""Modal dialog to remap Experiment (plant identifier) labels."""

from PyQt5 import QtCore, QtWidgets

HELP_TEXT = (
    "List of unique plant identifiers in this project. "
    "Rename them to fix typos or merge duplicates, or assign full genotype "
    "names after analyzing with short codes (e.g. 1, 2, 3)."
)

# Column widths shared by header and data rows
_CURRENT_WIDTH = 160
_COUNT_WIDTH = 120


class RemapIdentifiersDialog(QtWidgets.QDialog):
  def __init__(self, parent=None):
    super().__init__(parent)
    self.setWindowTitle("Remap plant identifiers")
    self.setModal(True)
    self.resize(520, 420)

    self._rows = []  # list of (old_name, QLineEdit)

    self.help_label = QtWidgets.QLabel(HELP_TEXT)
    self.help_label.setWordWrap(True)

    # Column titles (aligned with data rows below)
    header = QtWidgets.QWidget()
    header_layout = QtWidgets.QHBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.setSpacing(8)
    for text, width, stretch in (
        ("Current Identifier", _CURRENT_WIDTH, 0),
        ("Number of Plants", _COUNT_WIDTH, 0),
        ("New Identifier", None, 1),
    ):
      label = QtWidgets.QLabel(text)
      font = label.font()
      font.setBold(True)
      label.setFont(font)
      if width is not None:
        label.setFixedWidth(width)
      header_layout.addWidget(label, stretch)

    # Scrollable list of rows: current name | plant count | editable new name
    self._rows_host = QtWidgets.QWidget()
    self._rows_layout = QtWidgets.QVBoxLayout(self._rows_host)
    self._rows_layout.setContentsMargins(0, 0, 0, 0)
    self._rows_layout.setSpacing(6)
    self._rows_layout.addStretch()

    self.scroll = QtWidgets.QScrollArea()
    self.scroll.setWidgetResizable(True)
    self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    self.scroll.setWidget(self._rows_host)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    buttons.accepted.connect(self._on_accept)
    buttons.rejected.connect(self.reject)

    layout = QtWidgets.QVBoxLayout(self)
    layout.addWidget(self.help_label)
    layout.addWidget(header)
    layout.addWidget(self.scroll)
    layout.addWidget(buttons)

  def _clear_rows(self):
    self._rows = []
    while self._rows_layout.count() > 0:
      item = self._rows_layout.takeAt(0)
      widget = item.widget()
      if widget is not None:
        widget.deleteLater()
    self._rows_layout.addStretch()

  def populate(self, unique_counts):
    """Fill rows from {readable_experiment: plant_count}."""
    self._clear_rows()
    names = sorted(unique_counts.keys(), key=lambda s: (str(s).lower(), str(s)))

    for name in names:
      row = QtWidgets.QWidget()
      row_layout = QtWidgets.QHBoxLayout(row)
      row_layout.setContentsMargins(0, 0, 0, 0)
      row_layout.setSpacing(8)

      current_label = QtWidgets.QLabel(str(name))
      current_label.setFixedWidth(_CURRENT_WIDTH)
      current_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
      metrics = current_label.fontMetrics()
      current_label.setToolTip(str(name))
      current_label.setText(
          metrics.elidedText(str(name), QtCore.Qt.ElideRight, _CURRENT_WIDTH - 5)
      )

      count = unique_counts[name]
      count_text = "1 plant" if count == 1 else f"{count} plants"
      count_label = QtWidgets.QLabel(count_text)
      count_label.setFixedWidth(_COUNT_WIDTH)

      new_edit = QtWidgets.QLineEdit(str(name))

      row_layout.addWidget(current_label)
      row_layout.addWidget(count_label)
      row_layout.addWidget(new_edit, 1)

      # Insert before the trailing stretch
      self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
      self._rows.append((str(name), new_edit))

  def get_mapping(self):
    """Return {old: new} for rows where the new name differs after strip."""
    mapping = {}
    for old, edit in self._rows:
      new = edit.text().strip()
      if new and new != old:
        mapping[old] = new
    return mapping

  def validation_error(self):
    """Return an error message if any New identifier is blank, else None."""
    for old, edit in self._rows:
      if not edit.text().strip():
        return f"New identifier for '{old}' cannot be empty."
    return None

  def _on_accept(self):
    err = self.validation_error()
    if err:
      QtWidgets.QMessageBox.warning(self, "Invalid mapping", err)
      return
    self.accept()
