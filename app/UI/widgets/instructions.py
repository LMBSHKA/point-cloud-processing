from PySide6.QtWidgets import QMainWindow, QLabel

class SecondaryWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Управление")
        self.resize(500, 200)
        self.centralWidget = QLabel("""    Управление:
            K - фиксация экрана и перевод из свободного режима в режим выбора
            После этого можно с помощью ЛКМ выбрать прямоугольную область
            Или CTRL + ЛКМ для произвольной области
            C - Получение области точки с последующим уничтожением невыбранной области
            S - Сохранение выбранной области точек
            F - Переход в свободный режим""")
        self.setCentralWidget(self.centralWidget)