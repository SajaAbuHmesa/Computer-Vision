# 🔍 Real-Time Person Detection and Alert System

This project is a real-time person detection and notification system built using the YOLOv11 model. It processes video feeds to detect people and sends alerts via Telegram using a custom bot. It’s designed for smart surveillance and early-warning applications.
![image](https://github.com/user-attachments/assets/cd5944df-a951-468a-85ff-0dd2fddafb21)

---

## 🚀 Features

- 🎥 Real-time person detection using YOLOv11
- 🔍 Object detection via OpenCV and Ultralytics
- 🤖 Automated Telegram bot alerts upon detection
- 📦 Lightweight and easy to set up
- ⚡ CUDA/GPU support for faster performance

---

## 📁 Sample Video

To test the system, you can use the provided sample video:


```
https://drive.google.com/file/d/your-sample-video-id/view?usp=sharing](https://drive.google.com/file/d/11U0zwLPoA09_aJ_B_2SO1FHTbewklnCM/view?usp=sharing
```


---

## 🧠 Model

- **Model Used**: YOLOv11  
- **Weights Path**: `yolo/yolo11n.pt`

> Make sure to place your YOLOv11 weights in the correct directory.

---

## 💬 Telegram Alerts Setup

1. Talk to [BotFather](https://t.me/BotFather) on Telegram to create a bot.
2. Get your **Bot Token**.
3. Get your **Chat ID** using [@userinfobot](https://t.me/userinfobot).
4. Save your credentials in a `.env` file or directly in `telegram_settings.py`.

Example `telegram_settings.py`:

```python
bot_token = "YOUR_BOT_TOKEN"
chat_id = "YOUR_CHAT_ID"
```

## 🛠️ Installation
Clone the repository:
```
git clone https://github.com/yourusername/real-time-person-detection.git
cd real-time-person-detection
```
Install dependencies:
```
pip install -r requirements.txt
```
Run the detection script:
```

python main.py
```
## 📦 Requirements

* Python 3.8+
* PyTorch
* OpenCV
* NumPy
* Ultralytics
* Supervision
* python-telegram-bot
```
pip install torch opencv-python numpy ultralytics supervision python-telegram-bot
```
## 🧪 How It Works
The video stream is analyzed frame by frame.

If any person is detected (class_id = 0), the system:

Annotates the frame.

Sends a message to the Telegram bot once per detection session.

## 👤 Author
Saja Nasser Abu Hmesa
