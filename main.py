import cv2
import numpy as np

image_paths = [
    "data/road1.jpg",
    "data/road2.jpg",
    "data/road3.jpg",
    "data/road4.jpg"
]
video_path = "data/road_video.mp4"

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blur, 100, 200)
    roi = region_of_interest(edges)

    lines = detect_lines(roi)
    lane_lines = average_slope_intercept(frame, lines)
    line_image = draw_lines(frame, lane_lines)
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    combined = cv2.hconcat([
         frame,
         roi_bgr,
         result
    ])
    return combined

def region_of_interest(edges):

    height, width = edges.shape

    mask = np.zeros_like(edges)

    polygon = np.array([
    [
        (0, height),
        (width, height),
        (int(width * 0.55), int(height * 0.55)),
        (int(width * 0.45), int(height * 0.55))
    ]
     ], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    cropped = cv2.bitwise_and(edges, mask)

    return cropped
   
def detect_lines(cropped_edges):

    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=90,
        minLineLength=40,
        maxLineGap=5
    )

    return lines
def make_points(image, line):

    slope, intercept = line

    height = image.shape[0]

    y1 = height
    y2 = int(height * 0.6)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]
def average_slope_intercept(image, lines):

    left_fit = []
    right_fit = []

    if lines is None:
        return []

    for line in lines:

        x1, y1, x2, y2 = line.reshape(4)

        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    lane_lines = []

    if len(left_fit) > 0:
        left_avg = np.mean(left_fit, axis=0)
        lane_lines.append(make_points(image, left_avg))

    if len(right_fit) > 0:
        right_avg = np.mean(right_fit, axis=0)
        lane_lines.append(make_points(image, right_avg))

    return lane_lines
def draw_lines(image, lines):

    line_image = np.zeros_like(image)

    if lines is None:
        return line_image

    for line in lines:

        x1, y1, x2, y2 = line[0]

        cv2.line(
            line_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            8
        )

    return line_image


def process_images():
    for image_path in image_paths:
        image = cv2.imread(image_path)

        if image is None:
            print("Image not found:", image_path)
            continue

        combined = process_frame(image)

        cv2.imshow("Original | ROI | LANE DETECTION", combined)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def process_video():
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Video not found:", video_path)
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        combined = process_frame(frame)

        cv2.imshow("Video: Original | ROI | LANE DETECTION ", combined)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_images()
    process_video()
