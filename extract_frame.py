import cv2
import os
from natsort import natsorted   # ensures natural ordering (00001,00002,...)

def extract_frames(video_path, output_dir, H, W):
    """Extract frames from .MOV/.MP4 and save resized images.
       Skips frames already saved.
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file {video_path}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:  # End of video
            break

        # Construct output file path
        output_file = os.path.join(output_dir, f"{frame_count:06d}.jpg")

        # Skip if file already exists
        if os.path.exists(output_file):
            print(f"Skipping existing frame: {output_file}")
        else:
            # Resize frame
            resized = cv2.resize(frame, (W, H))

            # Save frame
            cv2.imwrite(output_file, resized)
            print(f"Saved: {output_file}")

        frame_count += 1

    cap.release()
    print(f"✨ Done. Total frames processed: {frame_count}")
    

def frames_to_video(image_dir, output_video, fps=60):
    """Create a video from extracted frames in a folder."""
    images = [img for img in os.listdir(image_dir) if img.lower().endswith((".jpg", ".png"))]
    if not images:
        print("❌ No images found in folder.")
        return

    images = natsorted(images)

    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for img_name in images:
        frame = cv2.imread(os.path.join(image_dir, img_name))
        video_writer.write(frame)
        print(f"Adding: {img_name}")

    video_writer.release()
    print(f"🎬 Video created successfully: {output_video}")


if __name__ == "__main__":
    # === User Configurable Parameters ===
    video_file = "2025-12-05 14-10-07.mkv"
    save_folder = "./0482"
    height = 540        # H
    width = 960        # W
    model = "DROID-SLAM"
    # extract_frames(video_file, save_folder, height, width)
    frames_folder = "./0482"
    FPS = 360
    output_video_file = f"IMG_0482_{FPS}fps_{model}.mp4"
    frames_to_video(frames_folder, output_video_file, fps=FPS)
