from pydub import AudioSegment
import os

def calculate_total_duration(folder_path):
    total_duration = 0

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            
            # Load the WAV file and get its duration in milliseconds
            audio = AudioSegment.from_wav(file_path)
            duration = len(audio)
            
            # Add the duration to the total duration
            total_duration += duration

    # Convert total duration to seconds
    total_duration_seconds = total_duration / 1000.0

    return total_duration_seconds

# Example usage
folder_path = "/nvme/uniamphion/se/Libritts_SE/wavs/train/CleanSpeech_training"
total_duration = calculate_total_duration(folder_path)

print(f"Total duration of WAV files in {folder_path}: {total_duration:.2f} seconds")
