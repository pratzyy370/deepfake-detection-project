import os
import subprocess
from tqdm import tqdm

# --- Configuration ---
# The script will look for these two folders in your project directory
INPUT_FOLDERS = ['TESTING VIDEOS(REAL)', 'TESTING VIDEOS(FAKE)']
# The converted, web-safe videos will be saved here
OUTPUT_FOLDER = 'converted_videos_for_demo'
# A list of video file extensions to look for
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv']

def convert_all_videos():
    """
    Finds videos in the specified input folders, converts them to a
    web-compatible H.264/AAC format, and saves them to organized subfolders.
    """
    print(f"--- Starting batch video conversion ---")

    if os.path.exists(OUTPUT_FOLDER):
        import shutil
        print(f"Removing old '{OUTPUT_FOLDER}' directory for a clean run...")
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)

    # Loop through each of the input folders (REAL and FAKE)
    for input_folder in INPUT_FOLDERS:
        if not os.path.exists(input_folder):
            print(f"\nWarning: Input folder '{input_folder}' not found. Skipping.")
            continue

        # Determine the output subfolder name ('real' or 'fake')
        if 'REAL' in input_folder.upper():
            output_subfolder = 'real'
        elif 'FAKE' in input_folder.upper():
            output_subfolder = 'fake'
        else:
            output_subfolder = 'other'
            
        final_output_path = os.path.join(OUTPUT_FOLDER, output_subfolder)
        os.makedirs(final_output_path, exist_ok=True)

        video_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]

        if not video_files:
            print(f"\nNo video files found in '{input_folder}'.")
            continue

        print(f"\nFound {len(video_files)} videos in '{input_folder}'. Starting conversion...")

        # Loop through each video in the current folder and convert it
        for filename in tqdm(video_files, desc=f"Converting '{input_folder}'"):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"web_{os.path.splitext(filename)[0]}.mp4"
            output_path = os.path.join(final_output_path, output_filename)
            
            command = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264', '-c:a', 'aac',
                '-y', output_path
            ]

            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                tqdm.write(f"  Error converting {filename}: {e.stderr.decode()}")
    
    print(f"\n✅ Conversion complete! All new videos are organized in the '{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    convert_all_videos()