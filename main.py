import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import time
import os
import pygame 
from collections import deque 
from datetime import datetime

# --- 設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_ROOT = os.path.join(BASE_DIR, 'New_Training_Data')
MODEL_PATH = os.path.join(BASE_DIR, 'janken_EF.pth')
AUDIO_PATH =  os.path.join(BASE_DIR, 'jankenpon_rhythm.wav')

# --- モデル定義 ---
INPUT_SIZE = 63
SEQ_LEN = 5
NUM_CLASSES = 3
CLASS_LABELS = ['gu', 'ch', 'pa']
PON_TIMING = 2.50

# --- アスキーアートのテキスト定義 ---
AA_TEXTS = {
    'gu': ( # プレイヤーが 'gu' -> 勝ち手は 'PA'
"                          @@",
"             @@        @    @",
"            @.  @.     @.    @.     @@@",
"            @.    @.   @.    @.    @.   @",
"            @.    @.   @.    @.   @.    @",
"            @.    @.   @.    @.  @.     @",
"            @.     @  @.     @. @.     @",
"            @.     @  @.     @.@.     @",
"            @.     @  @.      @      @    @@@",
"             @     @  @.     @.     @   @.   @",
"             @.     @@.      @.     @.  @    @",
"  @@@       @.      @.      @.      @ @.    @",
" @    @.    @.    @@@.    @.      @ @.    @",
"@       @   @.         @@@       @@.    @",
" @        @ @.               @@@        @",
"   @       @@.                    @     @",
"    @         @@                       @",
"     @            @                    @",
"      @            @                  @",
"      @             @                 @",
"       @              @               @",
"        @              @              @",
"         @                            @",
"          @                          @",
"           @                       @",
"             @                    @",
"              @                 @@",
"              @                  @",
"              @                  @",
"              @                  @",
"              @                  @",
"              @@@@@@@@@@@@",
"              Pa"
    ),
    'ch': ( # プレイヤーが 'ch' -> 勝ち手は 'GU'
"                         @@@@",
"                   @@@@.     @@",   
"                @@               @",
"         @@@@                    @",
"       @@                @         @",
"      @           @       @    @@@@",
"    @              @.       @@@.    @@",
"   @                 @     @            @",
"  @        @.        @.   @               @",
"  @         @.        @@@                 @",
"  @           @        @                   @",
" @             @        @          @      @",
"@               @.      @@.    @@@.     @",
"@.      @        @     @. @@@@         @",
"@.    @@         @.   @.  @              @",
" @       @      @ @@@  @               @",
"  @        @@@@.       @               @",
"  @                      @              @",
"   @                                    @",
"    @                                  @",
"     @                               @",
"       @@@                     @@@",
"           @@                    @",
"           @                      @",
"           @                      @",
"           @                      @",
"           @                      @",
"           @                      @",
"           @                      @",
"            @@@@@@@@@@@@@@",
"              Gu "
    ),
    'pa': ( # プレイヤーが 'pa' -> 勝ち手は 'CH'
"    @@@@.                  @@@@",
"  @       @                 @      @",      
"  @.        @.               @       @",     
"  @          @              @       @",     
"  @           @.            @        @",     
"   @.          @.          @.         @",    
"   @.           @.         @.         @",    
"     @.          @.        @.         @",    
"     @.          @.        @.         @",    
"       @.        @.       @.          @",    
"        @.        @.      @.         @",     
"         @.        @.     @.        @",      
"          @.        @.    @.        @",     
"           @.        @   @         @",     
"           @.        @.  @         @",       
"            @        @@@         @",      
"            @          @           @",
"         @@@@                     @",
"       @       @@.     @@@@@.   @",
"       @.          @@@.        @@@",
"     @.            @.               @@",
"     @           @                    @",
"   @@@@.       @                      @",
" @.       @@.   @                        @",
"@            @   @@@                    @",
"@             @.      @@@@@.            @",
"@               @.   @.  @                @",
"@.   @.          @. @.    @               @",
"@.    @.         @@.    @.                @",
"@      @.        @     @                 @",
" @      @@@@@@.     @                 @",
"  @                     @                @",
"   @                                    @",
"    @                                  @",
"     @                                @",
"       @                           @@",
"         @@@                      @",
"           @                        @",
"           @                        @",
"           @                        @",
"           @                        @",
"           @@@@@@@@@@@@@@@@",
"               CHOKI"
    )
}

# --- LSTMモデル ---
class HandGestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, num_classes=3):
        super(HandGestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 32).to(x.device)
        c0 = torch.zeros(1, x.size(0), 32).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# --- ユーティリティ関数 ---
def preprocess_landmarks(landmark_sequence_np):
    """ ランドマークの前処理 (相対座標化・正規化) """
    processed = []
    base = None
    for frame in landmark_sequence_np:
        if frame.sum() == 0:
            processed.append(np.zeros(63))
            continue
        
        landmarks = frame.reshape(-1, 3)
        if base is None: base = landmarks[0].copy()
        
        rel = landmarks - base
        scale = np.linalg.norm(rel[9]) # 中指付け根までの距離
        if scale < 1e-6: scale = 1.0
        
        processed.append((rel / scale).flatten())
    return np.array(processed, dtype=np.float32)

def create_aa_image(text_lines, width, height):
    """ アスキーアートを描画した画像を作成 """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    line_h = 10
    start_y = (height - (line_h * len(text_lines))) // 2 + line_h
    
    for i, line in enumerate(text_lines):
        cv2.putText(img, line, (10, start_y + i * line_h), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    return img

def save_feedback_data(sequence_np, correct_label):
    """ 
    間違った予測のデータを、正しいラベルのフォルダに保存する
    sequence_np: (SEQ_LEN, 63) の前処理済みデータ
    correct_label: 'gu', 'ch', 'pa' のいずれか
    """
    # 1. ラベルごとの保存先フォルダパスを作成 (例: .../new_training_data/pa)
    save_dir = os.path.join(FEEDBACK_ROOT, correct_label)
    
    # フォルダがなければ作成
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. ファイル名を生成 (被らないように日時を入れる)
    # 例: pa_feedback_20251031_102030.npy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{correct_label}_feedback_{timestamp}.npy"
    save_path = os.path.join(save_dir, filename)
    
    # 3. NumPy配列として保存
    np.save(save_path, sequence_np)
    print(f"学習データを保存しました: {save_path}")

# --- メイン処理 ---
def main():
    # デバイス・モデル準備
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = HandGestureLSTM(INPUT_SIZE).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model loaded.")
    except Exception as e:
        print(f"Model load error: {e}"); return
    
    #初回推論の遅延を防ぐためのウォームアップ
    print("Warming up model...")
    # ダミー入力 (Batch=1, SeqLen=5, InputSize=63)
    dummy_input = torch.zeros(1, SEQ_LEN, INPUT_SIZE).to(device)
    with torch.no_grad():
        model(dummy_input)
    print("Warm-up complete.")

    # MediaPipe準備
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils

    # 音声準備
    pygame.mixer.init()
    try:
        sound = pygame.mixer.Sound(AUDIO_PATH)
        print("Audio loaded.")
    except Exception as e:
        print(f"Audio load error: {e}"); sound = None

    # カメラ準備
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Camera error"); return
    
    # カメラ設定 (30fps固定)
    cap.set(cv2.CAP_PROP_FPS, 30)
    w, h = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # AA画像生成
    print("Generating AA images...")
    win_images = {k: create_aa_image(v, actual_w, actual_h) for k, v in AA_TEXTS.items()}
    
    # 変数初期化
    buffer = deque(maxlen=SEQ_LEN)
    for _ in range(SEQ_LEN): buffer.append(np.zeros(INPUT_SIZE))
    
    status = "Press SPACE"
    prediction = "N/A"
    is_capturing = False
    showing_result = False
    waiting_for_feedback = False
    start_time = 0
    result_img = None

    last_processed_seq = None
    
    print("Ready.")
    print("Controls:")
    print("  [SPACE]: ゲーム開始")
    print("  [SPACE]: Correct! (Go to next)")
    print("  [v]: Wrong! It was GU")
    print("  [b]: Wrong! It was CHOKI")
    print("  [n]: Wrong! It was PA")
    print("  [q]: Quit")

    # ループ
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        curr_time = time.time()

        # 1. MediaPipe処理
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        
        landmarks = np.zeros(INPUT_SIZE)
        if results.multi_hand_landmarks:
            lh = results.multi_hand_landmarks[0]
            lm_list = []
            for lm in lh.landmark: lm_list.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(lm_list)
            if not showing_result:
                mp_drawing.draw_landmarks(display, lh, mp_hands.HAND_CONNECTIONS)
        
        buffer.append(landmarks)

        # 2. キー入力
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): break
        
        # ゲーム開始 (フィードバック待ちでなければ)
        elif key == ord(' ') and not is_capturing and not waiting_for_feedback:
            status = "Starting..."
            is_capturing = True
            start_time = curr_time
            if sound: sound.play()
            
        # ★★★ フィードバック入力処理 ★★★
        elif waiting_for_feedback:
            if key == ord(' '): # 正解
                status = "Correct! Next..."
                waiting_for_feedback = False
                showing_result = False
                # (必要なら正解データも保存可能)
            
            elif key in [ord('v'), ord('b'), ord('n')]: # 訂正
                correct_label = ''
                if key == ord('v'): correct_label = 'gu'
                if key == ord('b'): correct_label = 'ch'
                if key == ord('n'): correct_label = 'pa'
                
                # データを保存
                if last_processed_seq is not None:
                    save_feedback_data(last_processed_seq, correct_label)
                    status = f"Saved as {correct_label}!"
                
                waiting_for_feedback = False
                showing_result = False

        # 3. ゲームロジック
        if is_capturing:
            elapsed = curr_time - start_time
            
            if elapsed >= PON_TIMING:
                # 推論実行
                status = "Processing..."
                seq = preprocess_landmarks(np.array(list(buffer)))
                last_processed_seq = seq
                tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    out = model(tensor)
                    idx = torch.max(out, 1)[1].item()
                    label = CLASS_LABELS[idx]
                    conf = torch.softmax(out, 1)[0][idx].item()
                    prediction = f"{label} ({conf*100:.0f}%)"

                # 結果表示準備
                result_img = win_images.get(label, np.zeros((actual_h, actual_w, 3), dtype=np.uint8))
                showing_result = True
                waiting_for_feedback = True # ★ フィードバック待ちへ移行 ★
                
                is_capturing = False
                status = "Was I correct? (space/v/b/n)"

        # 4. 結果描画
        if showing_result:
            cv2.convertScaleAbs(display, display, alpha=0.5, beta=0.0) # 背景暗く
            display = cv2.add(display, result_img) # AA合成


        # 5. ステータス表示
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (63, 192, 0), 2)
        cv2.putText(display, f"Your Hand: {prediction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Janken', display)

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()
    if sound: pygame.mixer.quit()
    hands.close()

if __name__ == "__main__":
    main()