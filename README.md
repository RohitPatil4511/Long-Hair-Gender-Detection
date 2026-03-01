# Long Hair Conditional Gender Detection System

## 📌 Project Overview

This project implements a conditional AI-based gender classification system that modifies predictions based on age and hair length logic.

The system integrates multiple computer vision models into a single intelligent decision pipeline.

---

## 🧠 Logic Design

### 🔹 Age between 20–30:
- Long Hair → Classified as Female
- Short Hair → Classified as Male

### 🔹 Age below 20 or above 30:
- Predict biological gender normally
- Hair length is ignored

---

## 🏗 System Architecture

- Face Detection (OpenCV)
- Age Prediction Model
- Gender Classification Model
- Hair Length Detection Model
- Conditional Decision Engine
- Flask-based GUI Interface

---

## 💻 Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- Flask
- HTML/CSS

---

## 🚀 Features

- Real-time image upload
- Face detection
- Age estimation
- Hair length classification
- Conditional gender logic
- Web-based graphical interface

---

## 📷 Application Workflow

1. User uploads image
2. Face is detected
3. Age is predicted
4. Hair length is analyzed
5. Conditional logic applied
6. Final result displayed in GUI

---

## 🎯 Purpose

This project demonstrates:

- Multi-model ML integration
- Rule-based AI logic implementation
- End-to-end deployment of computer vision system
- Backend–frontend ML integration

---

## 📌 Future Improvements

- Add confidence scores
- Improve model accuracy
- Deploy on cloud (Render / Railway)
- Add logging & analytics

---

## 👨‍💻 Author

Rohit Patil  
AI & ML Student