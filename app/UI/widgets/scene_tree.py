from __future__ import annotations

from dataclasses import dataclass
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem


@dataclass(frozen=True)
class SceneItem:
    obj_id: str
    name: str
    kind: str  # "cloud" / "model" / etc.
    path: str


class SceneTree(QTreeWidget):
    """
    Дерево проекта: Облака / Модели / Сравнения.
    Сейчас минимум: добавление облаков и клик по элементу.
    """
    item_selected = Signal(str)     # obj_id
    visibility_changed = Signal(str, bool)  # obj_id, visible

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setHeaderLabel("Дерево проекта")

        self._items: dict[str, SceneItem] = {}
        self._node_by_obj_id: dict[str, QTreeWidgetItem] = {}

        self.node_clouds = QTreeWidgetItem(["Облака точек"])
        self.node_models = QTreeWidgetItem(["Модели"])
        self.node_comp = QTreeWidgetItem(["Сравнения"])
        self.addTopLevelItems([self.node_clouds, self.node_models, self.node_comp])
        self.expandAll()

        self.itemClicked.connect(self._on_item_clicked)
        self.itemChanged.connect(self._on_item_changed)

    def add_cloud(self, item: SceneItem) -> None:
        self._items[item.obj_id] = item
        node = QTreeWidgetItem([item.name])
        node.setData(0, Qt.UserRole, item.obj_id)

        # checkbox видимости
        node.setFlags(node.flags() | Qt.ItemIsUserCheckable)
        node.setCheckState(0, Qt.Checked)

        self.node_clouds.addChild(node)
        self._node_by_obj_id[item.obj_id] = node
        self.expandAll()

    def add_model(self, item: SceneItem) -> None:
        """Добавляет объект в ветку «Модели»."""
        self._items[item.obj_id] = item

        node = QTreeWidgetItem([item.name])
        node.setData(0, Qt.UserRole, item.obj_id)

        node.setFlags(node.flags() | Qt.ItemIsUserCheckable)
        node.setCheckState(0, Qt.Checked)

        self.node_models.addChild(node)
        self._node_by_obj_id[item.obj_id] = node
        self.expandAll()

    def _on_item_clicked(self, item: QTreeWidgetItem) -> None:
        obj_id = item.data(0, Qt.UserRole)
        if isinstance(obj_id, str) and obj_id:
            self.item_selected.emit(obj_id)

    def _on_item_changed(self, item: QTreeWidgetItem) -> None:
        obj_id = item.data(0, Qt.UserRole)
        if not isinstance(obj_id, str) or not obj_id:
            return
        visible = item.checkState(0) == Qt.Checked
        self.visibility_changed.emit(obj_id, visible)
