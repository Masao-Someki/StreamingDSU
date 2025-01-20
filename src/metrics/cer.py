import jiwer
from espnet2.text.cleaner import TextCleaner


class CER:
    def __init__(self, type="whisper_basic"):
        self.cleaner = TextCleaner([type])
    
    def clean(self, text):
        cleaned = self.cleaner(text).strip()
        if cleaned == "":
            cleaned = "."
        return cleaned
    
    def compute_and_save(self, gts, hyps, save_path):
        wer = jiwer.cer(gts, hyps) * 100 # for score
        print(f"CER: {wer:.2f}%", flush=True)

        with open(f"{save_path}/cer", "w") as f:
            f.write(f"{wer:.2f}%")
        
        out = jiwer.process_words(gts, hyps)
        with open(f"{save_path}/cer_alignment", "w") as f:
            f.write(jiwer.visualize_alignment(out))
