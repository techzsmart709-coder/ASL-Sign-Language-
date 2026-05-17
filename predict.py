import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import warnings
import argparse
from classifier import GaussianNaiveBayes, KNN, StandardScaler
from utils import normalize_landmarks, extract_two_hands, extract_pairwise_distances
import os
import threading
import csv
import subprocess
import tkinter as tk
from tkinter import ttk

warnings.filterwarnings("ignore", category=UserWarning)

reload_model_flag = False

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), 
    (0, 5), (5, 6), (6, 7), (7, 8), 
    (5, 9), (9, 10), (10, 11), (11, 12), 
    (9, 13), (13, 14), (14, 15), (15, 16), 
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
]

def draw_landmarks_custom(frame, x_list, y_list):
    for connection in HAND_CONNECTIONS:
        pt1 = (x_list[connection[0]], y_list[connection[0]])
        pt2 = (x_list[connection[1]], y_list[connection[1]])
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    for i in range(len(x_list)):
        cv2.circle(frame, (x_list[i], y_list[i]), 4, (0, 0, 255), -1)

def get_model(method, k):
    scaler = StandardScaler()
    if method == 'bayes':
        model_path = 'bayes_model.npz'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} missing.")
            
        data = np.load(model_path, allow_pickle=True)
        model = GaussianNaiveBayes()
        model.classes_ = data['classes']
        model.priors_ = data['class_prior']
        model.means_ = data['class_mean']
        model.vars_ = data['class_var']
        
        scaler.mean_ = data['scaler_mean']
        scaler.scale_ = data['scaler_scale']
        return model, scaler
        
    elif method == 'rf':
        import joblib
        model_path = 'rf_model.joblib'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} missing.")
        data = joblib.load(model_path)
        model = data['model']
        scaler.mean_ = data['scaler_mean']
        scaler.scale_ = data['scaler_scale']
        return model, scaler
        
    elif method == 'knn':
        model_path = f'knn_model_k{k}.npz'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} missing.")
            
        data = np.load(model_path, allow_pickle=True)
        model = KNN(k=k)
        model.X_train_ = data['X_train']
        model.y_train_ = data['y_train']
        model.classes_ = data['classes']
        
        scaler.mean_ = data['scaler_mean']
        scaler.scale_ = data['scaler_scale']
        return model, scaler

def handle_report(features_list, wrong_prediction, wrong_shape, method, k):
    global reload_model_flag
    
    def on_radio_change(*args):
        lbl_sign.pack_forget()
        entry_sign.pack_forget()
        lbl_shape.pack_forget()
        entry_shape.pack_forget()
        btn_frame.pack_forget()
        
        issue = issue_var.get()
        if issue in ["Sign", "Both"]:
            lbl_sign.pack(pady=(5, 0))
            entry_sign.pack()
        if issue in ["Shape", "Both"]:
            lbl_shape.pack(pady=(5, 0))
            entry_shape.pack()
        btn_frame.pack(pady=15)
    
    def submit():
        global reload_model_flag
        issue = issue_var.get()
        c_sign = entry_sign.get().strip() if issue in ["Sign", "Both"] else ""
        c_shape = entry_shape.get().strip() if issue in ["Shape", "Both"] else ""
        
        try:
            with open("reported_errors.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([wrong_prediction, wrong_shape, issue, c_sign, c_shape] + features_list)
            
            if issue in ["Sign", "Both"] and c_sign:
                with open("asl_hand_landmarks.csv", "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(features_list + [c_sign])
                
                root.destroy()
                
                subprocess.run(["python", "main.py", "--train", "--method", method, "--k", str(k)])
                
                reload_model_flag = True
                return
        except Exception:
            pass
        root.destroy()
        
    def skip():
        root.destroy()

    root = tk.Tk()
    root.title("Report Detection Error")
    root.geometry("400x350")
    root.attributes("-topmost", True)
    
    ttk.Label(root, text=f"Flagged: '{wrong_prediction}' ({wrong_shape})", 
              font=("Arial", 11, "bold")).pack(pady=10)
    
    ttk.Label(root, text="What was incorrect?").pack()
    issue_var = tk.StringVar(value="Sign")
    issue_var.trace("w", on_radio_change)
    
    ttk.Radiobutton(root, text="Recognized Sign is wrong", variable=issue_var, value="Sign").pack(anchor="w", padx=80)
    ttk.Radiobutton(root, text="Hand State/Shape is wrong", variable=issue_var, value="Shape").pack(anchor="w", padx=80)
    ttk.Radiobutton(root, text="Both are wrong", variable=issue_var, value="Both").pack(anchor="w", padx=80)
    
    lbl_sign = ttk.Label(root, text="Correct Sign (e.g., 'hello'):")
    entry_sign = ttk.Entry(root, width=30)
    
    lbl_shape = ttk.Label(root, text="Correct Shape (e.g., 'Flat hand'):")
    entry_shape = ttk.Entry(root, width=30)
    
    btn_frame = ttk.Frame(root)
    ttk.Button(btn_frame, text="Submit Report & Train", command=submit).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Cancel / Skip", command=skip).pack(side=tk.LEFT, padx=5)
    
    on_radio_change()
    root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="Run ASL Prediction Webcam")
    parser.add_argument("--method", type=str, choices=['bayes', 'knn', 'rf'], required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    model, scaler = get_model(args.method, args.k)

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    confidence_threshold = 0.75
    spelled_text = ""
    last_char = ""
    last_added_time = 0
    cooldown_seconds = 2

    if args.method == 'bayes':
        method_name = "Naive Bayes"
    elif args.method == 'rf':
        method_name = "Random Forest"
    else:
        method_name = f"KNN (k={args.k})"
    
    start_time = time.time()
    
    global reload_model_flag

    while cap.isOpened():
        if reload_model_flag:
            model, scaler = get_model(args.method, args.k)
            reload_model_flag = False
            
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int((time.time() - start_time) * 1000)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if getattr(result, 'hand_landmarks', None):
            lm_list = extract_two_hands(result)
            if any(lm_list):
                lm_array = np.array([lm_list])
                lm_norm = normalize_landmarks(lm_array)
                lm_scaled = scaler.transform(lm_norm)
                
                SHAPE_MAP = {
                    "hello": "Flat hand",
                    "yes": "Fist (S-shape)",
                    "no": "3-finger pinch",
                    "eat": "Squished O-shape near mouth",
                    "drink": "C-shape near mouth",
                    "water": "W-shape at chin",
                    "sorry": "A-shape fist circles chest",
                    "where": "Index finger pointing up",
                    "who": "Hooked index finger at chin",
                    "fine": "5-shape spread at chest",
                    "play": "Two Y-shapes shaking",
                    "coffee": "Two fists grinding",
                    "book": "Flat hands opening",
                    "family": "Two F-shapes circling",
                    "house": "Flat hands touching (Roof)",
                    "boy": "C-shape at forehead",
                    "girl": "Thumb stroke on jaw",
                    "mother": "5-shape at chin",
                    "father": "5-shape at forehead",
                    "name": "H-shapes tapping",
                    "what": "Palms up and open",
                    "stop": "Flat hand chop",
                    "time": "Index tapping wrist",
                    "today": "Y-shapes bouncing down",
                    "tomorrow": "A-shape thumb moving forward",
                    "think": "Index tapping forehead",
                    "like": "8-shape pulling from chest",
                    "help": "A-hand on flat palm up",
                    "more": "Squished O-shapes tapping",
                    "finish": "5-shapes flicking outward",
                    "big": "L-shapes pulling apart",
                    "small": "Flat hands pushing together",
                    "hot": "Claw hand thrown from mouth",
                    "cold": "S-hands shaking",
                    "cat": "F-shapes pulling whiskers",
                    "read": "V-shape (2-fingers) scanning flat hand",
                    "talk": "4-shape (4-fingers up) at chin",
                    "bye": "Flat hand waving",
                    "love": "Two fists crossed over chest",
                    "boob": "Cupped hands at chest",
                    "sex": "X-hands tapping side of head"
                }
                
                from config import COMMON_WORDS, LIKELY_WORDS, LEAST_LIKELY_WORDS
                proba = model.predict_proba(lm_scaled)[0]
                
                common_indices = [i for i, c in enumerate(model.classes_) if c in COMMON_WORDS]
                likely_indices = [i for i, c in enumerate(model.classes_) if c in LIKELY_WORDS]
                least_likely_indices = [i for i, c in enumerate(model.classes_) if c in LEAST_LIKELY_WORDS]
                
                pred_index = -1
                
                if common_indices:
                    best_c_idx = common_indices[np.argmax(proba[common_indices])]
                    if proba[best_c_idx] >= confidence_threshold:
                        pred_index = best_c_idx
                        
                if pred_index == -1 and likely_indices:
                    best_l_idx = likely_indices[np.argmax(proba[likely_indices])]
                    if proba[best_l_idx] >= (confidence_threshold * 0.8):
                        pred_index = best_l_idx
                        
                if pred_index == -1 and least_likely_indices:
                    best_ll_idx = least_likely_indices[np.argmax(proba[least_likely_indices])]
                    if proba[best_ll_idx] >= (confidence_threshold * 0.6):
                        pred_index = best_ll_idx
                        
                if pred_index == -1:
                    pred_index = np.argmax(proba)
                    
                pred_char = str(model.classes_[pred_index])
                confidence = proba[pred_index]
                
                predicted_shape = SHAPE_MAP.get(pred_char, "Unknown Shape")

                current_time = time.time()
                if confidence >= confidence_threshold:
                    spelled_text = pred_char

                if args.method == "knn":
                    conf_str = f"{int(confidence * args.k)}/{args.k} votes"
                else:
                    conf_str = f"{confidence*100:.1f}%"
                    
                cv2.putText(frame, f'{pred_char} ({conf_str}) [{method_name}]',
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                            
                cv2.putText(frame, f'Shape: {predicted_shape}', (50, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                            
                h, w, _ = frame.shape
                cv2.putText(frame, "Press 'r' to report if prediction is wrong!", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            for hand_landmarks in result.hand_landmarks:
                h, w, _ = frame.shape
                x_list, y_list = [], []
                for lm in hand_landmarks:
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))
                draw_landmarks_custom(frame, x_list, y_list)

        cv2.putText(frame, f'Recognized: {spelled_text}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                    
        h, w, _ = frame.shape
        cv2.putText(frame, "Press 'Esc' to close", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("ASL Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            if 'lm_norm' in locals() and 'pred_char' in locals() and 'predicted_shape' in locals():
                norm_list = lm_norm[0].tolist()
                threading.Thread(target=handle_report, args=(norm_list, pred_char, predicted_shape, args.method, args.k), daemon=True).start()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
