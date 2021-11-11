import os
import argparse
import time

import librosa
import soundfile

from bytesep.inference import SeparatorWrapper

sample_rate = 44100  # Must be 44100 when using the downloaded checkpoints.

device = "cuda"  # "cuda" | "cpu"

vocals_separator = SeparatorWrapper(
    source_type='vocals',
    model=None,
    checkpoint_path=None,
    device=device,
)
accompaniment_separator = SeparatorWrapper(
    source_type='accompaniment',
    model=None,
    checkpoint_path=None,
    device=device,
)


def separate(audio_filename, audio_path):
    # Load audio.
    audio, fs = librosa.load(audio_path + audio_filename, sr=sample_rate, mono=False)

    if audio.ndim == 1:
        audio = audio[None, :]
        # (2, segment_samples)

    t1 = time.time()

    # Separate.
    sep_vocals_wav = vocals_separator.separate(audio)
    sep_accompaniment_wav = accompaniment_separator.separate(audio)

    sep_time = time.time() - t1

    # Write out audio
    output_filename = audio_filename.replace('.mp3', '.wav')
    sep_vocals_audio_path = audio_path + 'vocals/' + output_filename
    sep_accompaniment_audio_path = audio_path + 'accompaniment/' + output_filename

    soundfile.write(file=sep_vocals_audio_path, data=sep_vocals_wav.T, samplerate=sample_rate)
    soundfile.write(file=sep_accompaniment_audio_path, data=sep_accompaniment_wav.T, samplerate=sample_rate)

    print("Write out to {}, {}".format(sep_vocals_audio_path, sep_accompaniment_audio_path))
    print("Time: {:.3f}".format(sep_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--audios_path',
        type=str,
        default="/content/drive/MyDrive/musics/",
        help="Audio files folder",
    )
    args = parser.parse_args()

    audios_path = args.audios_path
    # read audio files in audios_path
    audio_file_paths = [file_path for file_path in os.listdir(audios_path) if os.path.isfile(audios_path + file_path)]
    # separate vocals and accomaniment

    for audio_file in audio_file_paths:
        separate(audio_file, audios_path)

