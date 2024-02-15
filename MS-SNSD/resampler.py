from pydub import AudioSegment
import os

def resample_flac_to_wav(input_path, output_path, target_sampling_rate=16000):
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Iterate through all files and subdirectories in the input path
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(".wav"):
                input_file_path = os.path.join(root, file)

                # Load the wav file
                audio = AudioSegment.from_file(input_file_path, format="wav")

                # Resample to the target sampling rate
                audio = audio.set_frame_rate(target_sampling_rate)

                # Create the output file path
                output_file_path = os.path.join(output_path, f"{os.path.splitext(file)[0]}.wav")

                # Export the resampled audio as a WAV file
                audio.export(output_file_path, format="wav")

                #print(f"Resampled and saved: {output_file_path}")

# Example usage
root_directory = "/nvme/uniamphion/tts/libritts/test-clean"
target_directory = "clean_libritts_test"

resample_flac_to_wav(root_directory, target_directory)
