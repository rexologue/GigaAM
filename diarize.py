from pathlib import Path
import torchaudio
from torchaudio.transforms import Resample
from nemo.collections.asr.models import SortformerEncLabelModel

src = Path("/root/audio/3844580337_0f49b1df58159bd58ea5f08fd4bbbe10_79036263700.mp3")
tmp = src.with_suffix(".mono16k.wav")

try:
    wav, sr = torchaudio.load(str(src))          # (channels, time)
    wav = wav.mean(dim=0, keepdim=True)          # mono -> (1, time)

    if sr != 16000:
        wav = Resample(sr, 16000)(wav)
        sr = 16000

    torchaudio.save(str(tmp), wav, sr)

    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
    diar_model.eval()

    diar_model.sortformer_modules.chunk_len = 340
    diar_model.sortformer_modules.chunk_right_context = 40
    diar_model.sortformer_modules.fifo_len = 40
    diar_model.sortformer_modules.spkcache_update_period = 300

    predicted_segments = diar_model.diarize(audio=[str(tmp)], batch_size=1)
    print(len(predicted_segments))
    for segment in predicted_segments[0]:
        print(segment, type(segment))

finally:
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
