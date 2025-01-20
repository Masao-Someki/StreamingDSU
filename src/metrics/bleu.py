from sacrebleu.metrics import BLEU as sacreBLEU, CHRF, TER
import re

class BLEU:
    def __init__(self,):
        self.bleu = sacreBLEU()
        self.chrf = CHRF()
        self.ter = TER()
    
    def clean(self, text):
        cleaned = re.sub(r'<[^>]+>', '', text)
        return cleaned.strip()
    
    def compute_and_save(self, gts, hyps, save_path):
        bleu_str = str(self.bleu.corpus_score(hyps, [gts]))
        chrf_str = str(self.chrf.corpus_score(hyps, [gts]))
        ter_str = str(self.ter.corpus_score(hyps, [gts]))

        score_str = "\n".join([bleu_str, chrf_str, ter_str])

        with open(f"{save_path}/score", "w") as f:
            f.write(score_str)
