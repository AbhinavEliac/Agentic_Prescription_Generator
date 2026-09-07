import sys
sys.modules['torchvision'] = None
sys.modules['torchaudio'] = None

import ctranslate2.converters.transformers as ct2_trans

out_dir = r"c:\Users\ADMIN\Downloads\rx_extractor_app_agentic\Whisper_Ayush_ct2"

print("Converting openai/whisper-large-v3-turbo to CTranslate2 INT8...")
converter = ct2_trans.TransformersConverter(
    model_name_or_path="openai/whisper-large-v3-turbo",
    low_cpu_mem_usage=True,
)
converter.convert(
    output_dir=out_dir,
    quantization="int8",
    force=True,
)
print("Conversion completed successfully!")
