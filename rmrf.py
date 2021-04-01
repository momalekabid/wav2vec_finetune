from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
import load_timits

load_timits.load()

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

print(timit["train"][0])