📁 ASL Custom Dataset – Setup Instructions
---


⚠️ Note:
If you're reading this file, it likely means you haven’t carefully read the main project README or are missing the dataset setup.

If you already have the dataset properly configured, you can ignore or delete this file.

📥 Dataset Download
---
Download the pre-filtered 1,000-word ASL Dataset from Kaggle:

👉 https://www.kaggle.com/datasets/aryanraj801/videos

📦 Extraction Instructions
Download the .zip file from Kaggle
Extract the contents
Ensure that all .mp4 video files are placed inside:
`dataset/videos/`

⚠️ Important:
---
Do not create extra nested folders inside `videos/`
All video files must be directly inside this directory
📄 Required File Structure

After setup, your folder should look like this:
```
dataset/
├── videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   ├── ...
└── WLASL_v0.3.json
```

🧠 Mapping File
---
Make sure the following file exists:

`dataset/WLASL_v0.3.json`

This file contains:

Word labels
Video mappings
Metadata required for training/inference

⚠️ The project will NOT work without this file.

✅ Final Checklist
---
Before running the project, verify:

 Dataset downloaded from Kaggle
 .mp4 files are inside `dataset/videos/`
 No extra subfolders inside `videos/`
 WLASL_v0.3.json exists in `dataset/`

🚨 Common Mistakes
---
❌ Placing videos in `dataset/videos/videos/`

❌ Missing WLASL_v0.3.json

❌ Keeping dataset zipped

❌ Renaming files or folders

💡 Tip
---
If you encounter errors like:

FileNotFoundError
Missing video paths
Dataset not loading

👉 99% of the time it's due to incorrect folder structure.

📬 Final Reminder
---
If your dataset is already working perfectly ✅
👉 You can safely ignore or delete this file
