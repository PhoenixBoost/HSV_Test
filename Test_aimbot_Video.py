import cv2
import numpy as np
import configparser
import os
from datetime import datetime

# **1. อ่านค่า Config จากไฟล์ Config_Video.ini**
config = configparser.ConfigParser()
config.read('Config_Video.ini')  # ใช้ไฟล์ Config_Video.ini

# อ่านค่าของ HSV Custom Lower และ Upper จากไฟล์ .ini
HSV_Custom_Lower = np.array([int(x) for x in config.get('HSV_Values', 'HSV_Custom_Lower').split(',')])
HSV_Custom_Upper = np.array([int(x) for x in config.get('HSV_Values', 'HSV_Custom_Upper').split(',')])

# อ่านค่า video_path และ undesired_colors_file_path จากไฟล์ .ini
video_path = config.get('Paths', 'video_path')
undesired_colors_file_path = '/Users/9phoomphi/Desktop/PJ_Code_edit/HSV_TEST/Out_Put/required_undesired_colors.txt'

undesired_colors_hsv = []

# **2. อ่านค่าสีที่ไม่ต้องการจากไฟล์ (required_undesired_colors.txt)**
try:
    with open(undesired_colors_file_path, 'r') as file:
        line = file.readline()
        if line.startswith("Undesired colors:"):
            colors_string = line.split(": ")[1]
            color_groups = colors_string.strip().split(";")
            for color_group in color_groups:
                color_str = color_group.strip("[]").split(",")
                if len(color_str) == 3:
                    h, s, v = map(int, color_str)
                    undesired_colors_hsv.append(np.array([h, s, v]))
        else:
            print(f"คำเตือน: รูปแบบไฟล์สีที่ไม่ต้องการอาจไม่ถูกต้อง: {undesired_colors_file_path}")
except FileNotFoundError:
    print(f"คำเตือน: ไม่พบไฟล์สีที่ไม่ต้องการ: {undesired_colors_file_path}")
except Exception as e:
    print(f"คำเตือน: ข้อผิดพลาดในการอ่านไฟล์สีที่ไม่ต้องการ: {e}")

if not undesired_colors_hsv:
    print("ไม่มีสีที่ไม่ต้องการถูกโหลดจากไฟล์ หรือไฟล์ไม่มี")

# **3. อ่านไฟล์ .webm หรือ .mp4**
cap = cv2.VideoCapture(video_path)

# ตรวจสอบว่าเปิดไฟล์วิดีโอได้หรือไม่
if not cap.isOpened():
    print(f"ไม่สามารถเปิดไฟล์วิดีโอได้ที่ {video_path}")
    exit()

# **4. เตรียมโฟลเดอร์ Output**
output_folder = 'Out_Put'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# **5. กำหนด codec และ VideoWriter สำหรับการบันทึกวิดีโอ**
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # เลือก codec สำหรับ .mp4 (สามารถเปลี่ยนเป็น 'webm' หรืออื่นๆ ได้)
output_video_path = os.path.join(output_folder, 'processed_video.mp4')
out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

if not out.isOpened():
    print("ไม่สามารถสร้าง VideoWriter ได้")
    exit()

# **6. เพิ่มการควบคุม FPS**
fps = cap.get(cv2.CAP_PROP_FPS)  # ใช้ FPS จากไฟล์วิดีโอ
delay = int(1000 / fps)  # คำนวณเวลา delay สำหรับแต่ละเฟรม

# **7. อ่านแต่ละเฟรมจากวิดีโอและประมวลผล**
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("ไม่สามารถอ่านเฟรมจากวิดีโอได้หรือถึงจุดสิ้นสุดของวิดีโอ")
        break  # ถ้าไม่มีเฟรมให้เล่นแล้ว
    
    # **8. ลดขนาดภาพเพื่อเพิ่มประสิทธิภาพ**
    frame_resized = cv2.resize(frame, (640, 360))  # ปรับขนาดภาพให้เล็กลง
    hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # **9. สร้าง Desired Color Mask**
    desired_color_mask = cv2.inRange(hsv_frame, HSV_Custom_Lower, HSV_Custom_Upper)

    # **10. สร้าง Undesired Color Mask Total**
    undesired_mask_total = np.zeros_like(desired_color_mask)
    for color in undesired_colors_hsv:
        lower_bound = np.array([max(0, color[0]-5), max(0, color[1]-5), max(0, color[2]-5)])
        upper_bound = np.array([min(179, color[0]+5), min(255, color[1]+5), min(255, color[2]+5)])
        undesired_color_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        undesired_mask_total = cv2.bitwise_or(undesired_mask_total, undesired_color_mask)

    # **11. สร้าง Final Mask**
    final_mask = cv2.bitwise_and(desired_color_mask, cv2.bitwise_not(undesired_mask_total))

    # **12. นำ Final Mask ไปใช้กับภาพต้นฉบับ**
    # ตรวจสอบให้แน่ใจว่า final_mask มีขนาดเท่ากับ frame
    if final_mask.shape[:2] != frame_resized.shape[:2]:
        final_mask_resized = cv2.resize(final_mask, (frame_resized.shape[1], frame_resized.shape[0]))
    else:
        final_mask_resized = final_mask

    masked_output = cv2.bitwise_and(frame_resized, frame_resized, mask=final_mask_resized)

    # **13. คำนวณความแม่นยำของสีที่ตรวจพบ**
    total_pixels = frame_resized.size // 3
    correct_detected_pixels = cv2.countNonZero(final_mask_resized)
    accuracy_percentage = (correct_detected_pixels / total_pixels) * 100

    # **14. คำนวณค่าการติดตามสี (Tracking Efficiency)**
    tracking_efficiency = (cv2.countNonZero(desired_color_mask) / total_pixels) * 100

    # **15. รวมภาพต้นฉบับและผลลัพธ์ที่ประมวลผล**
    mask_height, mask_width = final_mask_resized.shape[:2]
    combined_image_width = mask_width * 2
    combined_image_height = mask_height * 2

    combined_image = np.zeros((combined_image_height, combined_image_width, 3), dtype=np.uint8)

    # รวมภาพต้นฉบับและภาพที่ประมวลผล
    combined_image[0:mask_height, 0:mask_width] = frame_resized  # แสดงภาพต้นฉบับ
    combined_image[0:mask_height, mask_width:combined_image_width] = masked_output  # แสดงผลลัพธ์ที่ประมวลผล

    # **16. คำนวณสัดส่วนของรูปภาพ และ ขนาดตัวหนังสือ**
    base_image_width = 900.0
    image_scale_factor = combined_image_width / base_image_width
    base_font_scale = 0.5
    font_scale = base_font_scale * image_scale_factor

    # **17. แสดงผลลัพธ์ (ภาพรวมในหน้าต่างเดียว) พร้อมตัวหนังสือปรับขนาด**
    cv2.putText(combined_image, f'Accuracy: {accuracy_percentage:.2f}%', (10, mask_height + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(combined_image, f'Tracking Efficiency: {tracking_efficiency:.2f}%', (mask_width + 10, mask_height + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    # บันทึกผลลัพธ์ลงในไฟล์
    out.write(combined_image)

    # **18. แสดงผล**
    cv2.imshow('Original vs Processed', combined_image)

    # รอการกดปุ่ม 'q' เพื่อออกจากการแสดงผล
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# **19. ปิดการอ่านไฟล์และเขียนไฟล์**
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"วิดีโอที่ประมวลผลแล้วถูกบันทึกที่: {output_video_path}")
