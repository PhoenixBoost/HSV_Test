import cv2
import numpy as np
import sys
import configparser
import os
from datetime import datetime

# **1. อ่านค่า Config จากไฟล์ .ini**
config = configparser.ConfigParser()
config.read('Config_Image.ini')

# อ่าน path สำหรับรูปภาพและไฟล์สีที่ไม่ต้องการ
image_path = config.get('Paths', 'image_path')
undesired_colors_file_path = config.get('Paths', 'undesired_colors_file_path')

# อ่านค่าของ HSV Custom Lower และ Upper จากไฟล์ .ini
HSV_Custom_Lower = np.array([int(x) for x in config.get('HSV_Values', 'HSV_Custom_Lower').split(',')])
HSV_Custom_Upper = np.array([int(x) for x in config.get('HSV_Values', 'HSV_Custom_Upper').split(',')])

undesired_colors_hsv = []

# **2. อ่านค่าสีที่ไม่ต้องการจากไฟล์ (ถ้ามี)**
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

if undesired_colors_hsv:
    print("สีที่ไม่ต้องการที่อ่านจากไฟล์:")
    for color in undesired_colors_hsv:
        print(f"  HSV: {color}")
else:
    print("ไม่มีสีที่ไม่ต้องการถูกโหลดจากไฟล์ หรือไฟล์ไม่มี")

# **3. สร้างหน้าต่างแสดงผล**
combined_window_name = 'Combined Color Aimbot View'
cv2.namedWindow(combined_window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(combined_window_name, 900, 600)

# **4. โหลดรูปภาพ (จากไฟล์ที่ได้จาก Config)**
try:
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ไม่สามารถโหลดรูปภาพ: '{image_path}' ได้ โปรดตรวจสอบไฟล์")
        exit()
except Exception as e:
    print(f"ข้อผิดพลาดในการโหลดรูปภาพ: {e}")
    exit()

# **5. ประมวลผลภาพและสร้าง Mask**
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# **5.1 สร้าง Desired Color Mask**
desired_color_mask = cv2.inRange(hsv_frame, HSV_Custom_Lower, HSV_Custom_Upper)

# **5.2 สร้าง Undesired Color Mask Total**
undesired_mask_total = np.zeros_like(desired_color_mask)
for color in undesired_colors_hsv:
    lower_bound = np.array([max(0, color[0]-5), max(0, color[1]-5), max(0, color[2]-5)])
    upper_bound = np.array([min(179, color[0]+5), min(255, color[1]+5), min(255, color[2]+5)])
    undesired_color_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    undesired_mask_total = cv2.bitwise_or(undesired_mask_total, undesired_color_mask)

# **5.3 สร้าง Final Mask**
final_mask = cv2.bitwise_and(desired_color_mask, cv2.bitwise_not(undesired_mask_total))

# **5.4 นำ Final Mask ไปใช้กับภาพต้นฉบับ**
masked_output = cv2.bitwise_and(frame, frame, mask=final_mask)

# **6. คำนวณความแม่นยำของสีที่ตรวจพบ**
# การคำนวณความแม่นยำ (Precision) จาก Desired Color Mask และ Final Mask
total_pixels = frame.size // 3  # พิกเซลทั้งหมดในภาพ (3 = จำนวนช่องสี BGR)
correct_detected_pixels = cv2.countNonZero(final_mask)  # จำนวนพิกเซลที่ตรวจพบสีที่ตรงกับ Mask ที่ต้องการ
accuracy_percentage = (correct_detected_pixels / total_pixels) * 100

# **7. คำนวณค่าการติดตามสี (Tracking Efficiency)**
# การคำนวณการติดตามสีจาก Desired Color Mask และภาพต้นฉบับ
tracking_efficiency = (cv2.countNonZero(desired_color_mask) / total_pixels) * 100

# **8. แสดงผลลัพธ์การคำนวณ**
print(f"Accuracy of detected color: {accuracy_percentage:.2f}%")
print(f"Tracking efficiency: {tracking_efficiency:.2f}%")

# **9. รวมภาพ Mask ทั้งหมดในภาพเดียวเพื่อแสดงผล**
mask_height, mask_width = desired_color_mask.shape[:2]
combined_image_width = mask_width * 2
combined_image_height = mask_height * 2

combined_image = np.zeros((combined_image_height, combined_image_width, 3), dtype=np.uint8)

desired_color_mask_color = cv2.cvtColor(desired_color_mask, cv2.COLOR_GRAY2BGR)
undesired_color_mask_total_color = cv2.cvtColor(undesired_mask_total, cv2.COLOR_GRAY2BGR)
final_mask_color = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)

combined_image[0:mask_height, 0:mask_width] = desired_color_mask_color
combined_image[0:mask_height, mask_width:combined_image_width] = undesired_color_mask_total_color
combined_image[mask_height:combined_image_height, 0:mask_width] = final_mask_color
combined_image[mask_height:combined_image_height, mask_width:combined_image_width] = masked_output

# **10. คำนวณสัดส่วนของรูปภาพ และ ขนาดตัวหนังสือ**
base_image_width = 900.0 # ขนาดความกว้างของรูปภาพเริ่มต้นที่เราตั้งไว้
image_scale_factor = combined_image_width / base_image_width # สัดส่วนของรูปภาพ
base_font_scale = 0.5 # ขนาดตัวหนังสือเริ่มต้น
font_scale = base_font_scale * image_scale_factor # ขนาดตัวหนังสือใหม่ ปรับตามสัดส่วนรูปภาพ

# **11. แสดงผลลัพธ์ (ภาพรวมในหน้าต่างเดียว) พร้อมตัวหนังสือปรับขนาด**
cv2.putText(combined_image, f'Accuracy: {accuracy_percentage:.2f}%', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(combined_image, f'Tracking Efficiency: {tracking_efficiency:.2f}%', (mask_width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imshow(combined_window_name, combined_image)

# **12. สร้างโฟลเดอร์ Output ถ้ายังไม่มี**
output_folder = 'Out_Put'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# **13. ตั้งชื่อไฟล์ภาพ**
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_image_path = os.path.join(output_folder, f"result_{timestamp}.png")

# **14. บันทึกภาพผลลัพธ์**
cv2.imwrite(output_image_path, combined_image)
print(f"ผลลัพธ์ภาพถูกบันทึกไว้ที่: {output_image_path}")

# **15. รอรับการกดปุ่ม (F9 หรือ ปุ่ม q เพื่อออก)**
running = True
while running:
    key = cv2.waitKey(1)
    if key == 133:  # ปุ่ม F9
        print("Exit program by F9 key")
        running = False
    elif key == ord('q'):  # ปุ่ม q
        print("Exit program by pressing 'q' key")
        running = False
    elif key == -1:
        pass  # ปิดหน้าต่าง

    if not running:
        break

# **16. ปิดโปรแกรมและหน้าต่างทั้งหมด**
cv2.destroyAllWindows()
