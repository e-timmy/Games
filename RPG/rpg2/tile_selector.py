import os
import sys
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel, QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget, QGridLayout
from PIL import Image  # Pillow library for image processing

class TileSelector(QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.selected_tiles = {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Tile Selector')
        main_layout = QVBoxLayout()

        # Create scroll area for the image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QGridLayout(scroll_content)

        # Open and display the image
        img = Image.open(self.image_path)
        width, height = img.size
        tiles_x = width // 16
        tiles_y = height // 16

        for y in range(tiles_y):
            for x in range(tiles_x):
                left = x * 16
                top = y * 16
                right = left + 16
                bottom = top + 16
                tile = img.crop((left, top, right, bottom))

                # Convert PIL Image to QPixmap
                qim = QImage(tile.tobytes("raw", "RGB"), tile.size[0], tile.size[1], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qim)

                # Create clickable label for each tile
                label = ClickableLabel(pixmap, x, y)
                label.clicked.connect(self.tile_clicked)
                scroll_layout.addWidget(label, y, x)

        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # Add save button
        save_button = QPushButton('Save Selected Tiles')
        save_button.clicked.connect(self.save_tiles)
        main_layout.addWidget(save_button)

        self.setLayout(main_layout)

    def tile_clicked(self, x, y):
        dialog = TileNameDialog(x, y)
        if dialog.exec_() == QDialog.Accepted:
            tile_name = dialog.get_name()
            if tile_name:
                self.selected_tiles[(x, y)] = tile_name
                print(f"Tile ({x}, {y}) named: {tile_name}")

    def save_tiles(self):
        if not self.selected_tiles:
            print("No tiles selected.")
            return

        # Create subdirectory for saving tiles
        base_dir = os.path.dirname(self.image_path)
        file_name = os.path.splitext(os.path.basename(self.image_path))[0]
        save_dir = os.path.join(base_dir, file_name + "_tiles")
        os.makedirs(save_dir, exist_ok=True)

        # Open the original image
        img = Image.open(self.image_path)

        # Save selected tiles
        for (x, y), name in self.selected_tiles.items():
            left = x * 16
            top = y * 16
            right = left + 16
            bottom = top + 16
            tile = img.crop((left, top, right, bottom))

            tile_path = os.path.join(save_dir, f"{name}.png")
            tile.save(tile_path)
            print(f"Tile saved as {tile_path}")

        print("All selected tiles have been saved.")
        self.close()


class ClickableLabel(QLabel):
    def __init__(self, pixmap, x, y):
        super().__init__()
        self.setPixmap(pixmap.scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.x = x
        self.y = y

    def mousePressEvent(self, event):
        self.clicked.emit(self.x, self.y)

    clicked = pyqtSignal(int, int)

class SingleKeyPressLineEdit(QLineEdit):
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace:
            # Handle backspace separately
            self.setText(self.text()[:-1])
        else:
            # For other keys, append the text
            self.setText(self.text() + event.text())


class TileNameDialog(QDialog):
    def __init__(self, x, y):
        super().__init__()
        self.setWindowTitle(f'Name Tile ({x}, {y})')
        self.setModal(True)  # Make the dialog modal

        layout = QVBoxLayout(self)

        self.nameInput = QLineEdit(self)
        self.nameInput.setPlaceholderText("Enter tile name")
        layout.addWidget(self.nameInput)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_name(self):
        return self.nameInput.text()

def break_tiles():
    # Ask for the input file
    input_file = 'assets/town_rpg_pack/graphics/elements/basictiles.png'

    # Check if file exists
    if not os.path.exists(input_file):
        print("Error: File does not exist.")
        return

    app = QApplication(sys.argv)
    selector = TileSelector(input_file)
    selector.show()
    app.exec_()

if __name__ == "__main__":
    break_tiles()
    # main()