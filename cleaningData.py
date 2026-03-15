import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import shutil

# --- 設定 ---
# 1. チェック対象のデータフォルダ（ここにあるファイルを順に見ます）
CHECK_TARGET_DIRS = [
    'New_Training_Data1207/gu',
    'New_Training_Data1207/ch',
    'New_Training_Data1207/pa',
]

# 2. 振り分け先のルートフォルダ（この下に gu, ch, pa フォルダがある前提）
# 基本的に CHECK_TARGET_DIRS の親フォルダと同じでOKです
DEST_ROOT = 'newer_training_data' 

# 3. 削除データの移動先
TRASH_DIR = 'deleted_data'

# MediaPipeの手の接続情報
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

class ManualDataSorter:
    def __init__(self):
        self.action = None 

    def on_key(self, event):
        """ キーボード入力時の処理 """
        # --- 振り分け操作 ---
        if event.key == 'g':
            self.action = 'move_gu'
            plt.close()
        elif event.key == 'c':
            self.action = 'move_ch'
            plt.close()
        elif event.key == 'p':
            self.action = 'move_pa'
            plt.close()
            
        # --- その他の操作 ---
        elif event.key == 'd': # Delete (Trash)
            self.action = 'delete'
            plt.close()
        elif event.key == ' ' or event.key == 'enter': # Keep (その場に残す)
            self.action = 'keep'
            plt.close()
        elif event.key == 'q': # Quit
            self.action = 'quit'
            plt.close()

    def visualize_and_decide(self, file_path, current_idx, total_files):
        self.action = None 

        try:
            data = np.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return 'error'

        if data.size == 0: return 'delete'
        num_frames = data.shape[0]
        landmarks = data.reshape(num_frames, 21, 3)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        file_name = os.path.basename(file_path)
        parent_dir = os.path.basename(os.path.dirname(file_path)) # 現在のフォルダ名
        
        title_text = (
            f"[{current_idx}/{total_files}] Current: {parent_dir}/{file_name}\n"
            f"[g]:Move to GU, [c]:Move to CH, [p]:Move to PA\n"
            f"[Space]:Keep, [d]:Delete, [q]:Quit"
        )
        
        # 視点・範囲設定
        ax.view_init(elev=-90, azim=-90)
        limit = 1.0 # ★表示範囲を広げました
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        scat = ax.scatter([], [], [], c='r', marker='o')
        lines = [ax.plot([], [], [], 'b-')[0] for _ in range(len(HAND_CONNECTIONS))]

        def update(frame_idx):
            ax.set_title(f"{title_text}\nFrame: {frame_idx}/{num_frames}")
            current_pose = landmarks[frame_idx]
            xs, ys, zs = current_pose[:, 0], current_pose[:, 1], current_pose[:, 2]

            scat._offsets3d = (xs, ys, zs)
            for line, (start, end) in zip(lines, HAND_CONNECTIONS):
                line.set_data([xs[start], xs[end]], [ys[start], ys[end]])
                line.set_3d_properties([zs[start], zs[end]])
            return scat, lines

        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)
        fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print(f"Checking: {file_name} ... ", end='', flush=True)
        plt.show()

        return self.action

def move_file(file_path, dest_dir_name):
    """ ファイルを指定されたフォルダへ移動する """
    try:
        file_name = os.path.basename(file_path)
        
        # 移動先ディレクトリパス
        if dest_dir_name == 'TRASH':
            target_dir = TRASH_DIR
        else:
            target_dir = os.path.join(DEST_ROOT, dest_dir_name)
        
        os.makedirs(target_dir, exist_ok=True)
        
        # 移動先パス
        dest_path = os.path.join(target_dir, file_name)

        # 既に同じ場所にいる場合は何もしない
        if os.path.abspath(os.path.dirname(file_path)) == os.path.abspath(target_dir):
            print(f"-> [Kept] (Already in {dest_dir_name})")
            return

        # 同名ファイル回避
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(file_name)
            import time
            timestamp = str(int(time.time()))
            dest_path = os.path.join(target_dir, f"{base}_{timestamp}{ext}")

        shutil.move(file_path, dest_path)
        print(f"-> [Moved] to {dest_dir_name}/")
        
    except Exception as e:
        print(f"-> Error moving: {e}")

def main():
    # 1. ファイルリスト作成
    all_files = []
    for d in CHECK_TARGET_DIRS:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, '*.npy'))
            all_files.extend(files)
    
    if not all_files:
        print("ファイルが見つかりません。")
        return

    print(f"合計 {len(all_files)} 個のファイルを整理します。")
    print("-" * 50)

    sorter = ManualDataSorter()

    for i, file_path in enumerate(all_files, 1):
        # ファイルが既に移動済み（手前の操作で削除など）でないか確認
        if not os.path.exists(file_path): continue

        action = sorter.visualize_and_decide(file_path, i, len(all_files))

        if action == 'move_gu':
            move_file(file_path, 'gu')
        elif action == 'move_ch':
            move_file(file_path, 'ch')
        elif action == 'move_pa':
            move_file(file_path, 'pa')
        elif action == 'delete':
            move_file(file_path, 'TRASH')
        elif action == 'keep':
            print("-> [Kept]")
        elif action == 'quit':
            print("\n中断しました。")
            break
        elif action == 'error':
            print("-> Skip")

    print("完了しました。")

if __name__ == "__main__":
    main()