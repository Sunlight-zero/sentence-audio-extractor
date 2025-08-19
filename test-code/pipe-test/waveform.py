import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import wave
import numpy as np
import pygame
import threading
import time

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AudioWaveformPlayer:
    """
    一个音频播放器应用，可以显示波形图并实时标记播放位置。
    """
    def __init__(self, root):
        self.root = root
        self.root.title("音频波形播放器")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # --- 初始化变量 ---
        self.filepath = None
        self.audio_data = None
        self.framerate = 0
        self.duration = 0
        self.playback_line = None
        self.is_playing = False
        self.is_paused = False
        self.stop_thread = False # 用于停止更新线程的标志

        # --- 初始化 Pygame Mixer ---
        try:
            pygame.mixer.init()
        except pygame.error as e:
            messagebox.showerror("Pygame 错误", f"无法初始化音频设备: {e}\n请确保您有可用的音频输出设备。")
            self.root.destroy()
            return

        # --- 创建 GUI 组件 ---
        self._create_widgets()

    def _create_widgets(self):
        """创建并布局所有 GUI 组件。"""
        # --- 顶部控制面板 ---
        control_frame = tk.Frame(self.root, bg="#dcdcdc", pady=10)
        control_frame.pack(fill=tk.X, side=tk.TOP)

        self.open_button = ttk.Button(control_frame, text="打开音频文件", command=self.open_file)
        self.open_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.play_pause_button = ttk.Button(control_frame, text="▶ 播放", command=self.toggle_play_pause, state=tk.DISABLED)
        self.play_pause_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = ttk.Button(control_frame, text="■ 停止", command=self.stop_playback, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.filename_label = ttk.Label(control_frame, text="尚未选择文件", background="#dcdcdc", anchor=tk.W)
        self.filename_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # --- Matplotlib 波形图画布 ---
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.fig.patch.set_facecolor('#ffffff') # 设置图形背景色
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#ffffff') # 设置坐标轴背景色
        self.ax.set_title("音频波形图")
        self.ax.set_xlabel("时间 (秒)")
        self.ax.set_ylabel("振幅")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.draw()

    def open_file(self):
        """打开一个 .wav 文件，加载数据并绘制波形。"""
        # 仅支持 .wav 文件
        filepath = filedialog.askopenfilename(
            title="请选择一个 WAV 文件",
            filetypes=[("WAV files", "*.wav")]
        )
        if not filepath:
            return

        try:
            with wave.open(filepath, 'rb') as wf:
                self.filepath = filepath
                self.framerate = wf.getframerate()
                n_frames = wf.getnframes()
                self.duration = n_frames / float(self.framerate)
                
                # 读取音频数据并转换为 NumPy 数组
                frames = wf.readframes(n_frames)
                self.audio_data = np.frombuffer(frames, dtype=np.int16)

                # 如果是立体声，只取一个声道用于显示
                if wf.getnchannels() == 2:
                    self.audio_data = self.audio_data[::2]

            self.plot_waveform()
            
            # 加载音频到 Pygame Mixer
            pygame.mixer.music.load(self.filepath)
            
            # 更新UI
            self.filename_label.config(text=self.filepath.split('/')[-1])
            self.play_pause_button.config(state=tk.NORMAL, text="▶ 播放")
            self.stop_button.config(state=tk.NORMAL)
            self.is_playing = False
            self.is_paused = False

        except Exception as e:
            messagebox.showerror("错误", f"无法打开或处理文件: {e}")
            self.filepath = None

    def plot_waveform(self):
        """使用 Matplotlib 绘制波形图。"""
        self.ax.clear()
        time_axis = np.linspace(0, self.duration, num=len(self.audio_data))
        self.ax.plot(time_axis, self.audio_data, linewidth=0.5, color="#007acc")
        self.ax.set_title("音频波形图")
        self.ax.set_xlabel("时间 (秒)")
        self.ax.set_ylabel("振幅")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        # 创建可移动的播放位置竖线
        self.playback_line = self.ax.axvline(x=0, color='r', linestyle='-', linewidth=1.5)
        
        self.canvas.draw()

    def toggle_play_pause(self):
        """切换播放和暂停状态。"""
        if not self.is_playing:
            # 开始播放
            pygame.mixer.music.play()
            self.is_playing = True
            self.is_paused = False
            self.play_pause_button.config(text="❚❚ 暂停")
            
            # 启动一个新线程来更新播放线
            self.stop_thread = False
            self.update_thread = threading.Thread(target=self.update_playback_line, daemon=True)
            self.update_thread.start()
        elif self.is_playing and not self.is_paused:
            # 暂停
            pygame.mixer.music.pause()
            self.is_paused = True
            self.play_pause_button.config(text="▶ 继续")
        else:
            # 从暂停中恢复
            pygame.mixer.music.unpause()
            self.is_paused = False
            self.play_pause_button.config(text="❚❚ 暂停")

    def stop_playback(self):
        """停止播放并重置。"""
        if self.filepath:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.is_paused = False
            self.stop_thread = True # 发送停止信号给更新线程
            self.play_pause_button.config(text="▶ 播放")
            
            # 重置播放线到起点
            if self.playback_line:
                self.playback_line.set_xdata([0])
                self.canvas.draw_idle()

    def update_playback_line(self):
        """在一个独立的线程中循环更新播放线的位置。"""
        while pygame.mixer.music.get_busy() and not self.stop_thread:
            if not self.is_paused:
                # 获取播放的毫秒数并转换为秒
                current_time = pygame.mixer.music.get_pos() / 1000.0
                
                # 更新 Matplotlib 竖线的位置
                self.playback_line.set_xdata([current_time])
                
                # 在主线程中安全地重绘 canvas
                self.root.after(0, self.canvas.draw_idle)
            
            time.sleep(0.05) # 每 50 毫秒更新一次

        # 播放结束后，重置状态
        if not self.stop_thread:
            self.root.after(0, self.stop_playback)


    def on_closing(self):
        """在关闭窗口时，清理资源。"""
        self.stop_playback()
        pygame.mixer.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioWaveformPlayer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing) # 确保关闭窗口时调用 on_closing
    root.mainloop()
