from stable_audio_3 import StableAudioModel
import soundfile as sf

print("Loading model...")

model = StableAudioModel.from_pretrained("small-sfx")

print("Generating audio...")

audio = model.generate(
    prompt="helicopter flyng over a city",
    duration=5,
)

print("Audio shape:", audio.shape)

# Remove batch dimension
audio = audio[0]

# Convert from [channels, samples] to [samples, channels]
audio = audio.transpose(1, 0)

print("Saving audio...")

sf.write(
    "helicopter_flying_over_a_city.wav",
    audio,
    48000
)

print("Saved helicopter_flying_over_a_city.wav")