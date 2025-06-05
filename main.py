import PyQt5
from PyQt5 import QtWidgets
from PyQt5 import uic
import sys
import play

class GameWindow:

    def __init__(
        self, 
        main_interface_path = './ui/main_interface.ui', 
        game_introduction_path='./ui/game_introduction.ui',
        diffculty_change_path = './ui/diffculty_change.ui',
        win_step=200
    ):
        self.window = uic.loadUi(main_interface_path)
        self.game_introduction_window = uic.loadUi(game_introduction_path)
        self.diffculty_window = uic.loadUi(diffculty_change_path)
        self.win_step = win_step
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.window)
        self.stack.addWidget(self.diffculty_window)
        self.frame_time = 0.1
        # 主菜单逻辑
        self.window.start_game.clicked.connect(self.on_start_game_clicked)
        self.window.auto_mode.clicked.connect(self.on_auto_mode_clicked)
        self.window.diffculty_option.clicked.connect(self.on_diffculty_change_clicked)
        # 游戏介绍逻辑
        self.window.game_introduction.clicked.connect(self.on_game_introduction_clicked)
        #难易度选择逻辑
        self.diffculty_window.hard.clicked.connect(lambda : self.on_diffculty_selected_clicked('hard'))
        self.diffculty_window.middle.clicked.connect(lambda : self.on_diffculty_selected_clicked('middle'))
        self.diffculty_window.easy.clicked.connect(lambda : self.on_diffculty_selected_clicked('easy'))

        self.stack.show()

    def on_start_game_clicked(self):
        self.window.hide()
        step = play.start_game(True, True, frame_time=self.frame_time)
        messagebox = QtWidgets.QMessageBox()
        if step == self.win_step:
            messagebox.setText(f'You win!')
        else:
            messagebox.setText(f'You failed with step {step}')
        messagebox.exec_()
        self.window.show()
    
    def on_auto_mode_clicked(self):
        self.window.hide()
        step = play.start_game(False,False, frame_time=self.frame_time)
        messagebox = QtWidgets.QMessageBox()
        if step == self.win_step:
            messagebox.setText(f'You win!')
        else:
            messagebox.setText(f'You failed with step {step}')
        messagebox.exec_()
        self.window.show()
    
    def on_game_introduction_clicked(self):
        self.window.hide()
        # 显示介绍页面
        self.game_introduction_window.exec_()
        #回到主菜单
        self.window.show()

    def on_diffculty_change_clicked(self):
        self.stack.setCurrentWidget(self.diffculty_window)

    def on_diffculty_selected_clicked(self, diffculty):
        if diffculty == 'hard':
            self.frame_time = 0.05
        elif diffculty == 'middle':
            self.frame_time = 0.1
        else:
            self.frame_time = 0.15
        self.stack.setCurrentWidget(self.window)

        
app = QtWidgets.QApplication(sys.argv)
window = GameWindow()


app.exec_()