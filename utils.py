import json
import numpy as np

def extract_landmarks(results):
    landmark_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                landmark_list.append({
                    "id": i,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                })
    return landmark_list

def save_landmarks_to_json(landmarks, filename='hand_landmarks.json'):
    with open(filename, 'w') as f:
        json.dump(landmarks, f, indent=2)

def extract_two_hands(detection_result):
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63
    
    if getattr(detection_result, 'hand_landmarks', None):
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness = detection_result.handedness[idx][0].category_name
            lm_list = []
            for lm in hand_landmarks:
                lm_list.extend([lm.x, lm.y, lm.z])
                
            if handedness == "Left":
                left_hand = lm_list
            elif handedness == "Right":
                right_hand = lm_list
                
    return left_hand + right_hand

def normalize_landmarks(X):
    X_out = np.copy(X)
    
    if X_out.ndim == 1:
        if np.any(X_out[:63]): 
            wrist_x, wrist_y, wrist_z = X_out[0], X_out[1], X_out[2]
            X_out[0:63:3] -= wrist_x
            X_out[1:63:3] -= wrist_y
            X_out[2:63:3] -= wrist_z
            max_dist = np.max(np.abs(X_out[0:63]))
            if max_dist > 0:
                X_out[0:63] /= max_dist
            
        if np.any(X_out[63:]):
            wrist_x, wrist_y, wrist_z = X_out[63], X_out[64], X_out[65]
            X_out[63::3] -= wrist_x
            X_out[64::3] -= wrist_y
            X_out[65::3] -= wrist_z
            max_dist = np.max(np.abs(X_out[63:]))
            if max_dist > 0:
                X_out[63:] /= max_dist
    else:
        left_mask = np.any(X_out[:, :63], axis=1)
        if np.any(left_mask):
            wrist_lx = X_out[left_mask, 0:1]
            wrist_ly = X_out[left_mask, 1:2]
            wrist_lz = X_out[left_mask, 2:3]
            X_out[left_mask, 0:63:3] -= wrist_lx
            X_out[left_mask, 1:63:3] -= wrist_ly
            X_out[left_mask, 2:63:3] -= wrist_lz
            max_dists = np.max(np.abs(X_out[left_mask, :63]), axis=1, keepdims=True)
            X_out[left_mask, :63] /= (max_dists + 1e-8)
            
        right_mask = np.any(X_out[:, 63:], axis=1)
        if np.any(right_mask):
            wrist_rx = X_out[right_mask, 63:64]
            wrist_ry = X_out[right_mask, 64:65]
            wrist_rz = X_out[right_mask, 65:66]
            X_out[right_mask, 63::3] -= wrist_rx
            X_out[right_mask, 64::3] -= wrist_ry
            X_out[right_mask, 65::3] -= wrist_rz
            max_dists = np.max(np.abs(X_out[right_mask, 63:]), axis=1, keepdims=True)
            X_out[right_mask, 63:] /= (max_dists + 1e-8)
            
    return X_out

def extract_pairwise_distances(X_norm):
    is_1d = X_norm.ndim == 1
    if is_1d:
        X_norm = X_norm.reshape(1, -1)
        
    num_samples = X_norm.shape[0]
    out = np.zeros((num_samples, 420))
    
    for k in range(num_samples):
        left_coords = X_norm[k, :63].reshape(21, 3)
        if np.any(left_coords):
            idx = 0
            for i in range(21):
                for j in range(i+1, 21):
                    out[k, idx] = np.linalg.norm(left_coords[i] - left_coords[j])
                    idx += 1
                    
        right_coords = X_norm[k, 63:].reshape(21, 3)
        if np.any(right_coords):
            idx = 210
            for i in range(21):
                for j in range(i+1, 21):
                    out[k, idx] = np.linalg.norm(right_coords[i] - right_coords[j])
                    idx += 1
                    
    max_vals = np.max(out, axis=1, keepdims=True) + 1e-8
    out /= max_vals
    
    return out[0] if is_1d else out
