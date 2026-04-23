import argparse
import os
import cv2


def extract_frames(video_path: str, output_dir: str, image_format: str = "jpg") -> int:
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pad = max(6, len(str(total)))

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        filename = os.path.join(output_dir, f"frame_{count:0{pad}d}.{image_format}")
        cv2.imwrite(filename, frame)
        count += 1

    cap.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Save every frame of a video into a folder.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("-o", "--output", default="frames", help="Output folder (default: frames)")
    parser.add_argument("-f", "--format", default="jpg", choices=["jpg", "png", "bmp"], help="Image format")
    args = parser.parse_args()

    saved = extract_frames(args.video, args.output, args.format)
    print(f"Saved {saved} frames to {args.output}")


if __name__ == "__main__":
    main()
