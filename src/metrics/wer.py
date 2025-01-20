import jiwer
from espnet2.text.cleaner import TextCleaner


class WER:
    def __init__(self, type="whisper_basic"):
        self.cleaner = TextCleaner([type])
    
    def clean(self, text):
        cleaned = self.cleaner(text).strip()
        if cleaned == "":
            cleaned = "."
        return cleaned
    
    def compute_and_save(self, gts, hyps, save_path):
        wer = jiwer.wer(gts, hyps) * 100 # for score
        print(f"WER: {wer:.2f}%", flush=True)

        with open(f"{save_path}/wer", "w") as f:
            f.write(f"{wer:.2f}%")
        
        out = jiwer.process_words(gts, hyps)
        with open(f"{save_path}/wer_alignment", "w") as f:
            f.write(jiwer.visualize_alignment(out))
