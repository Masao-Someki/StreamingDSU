from itertools import groupby

from flask import Flask, request, jsonify
import torch.nn as nn

from espnet2.bin.mt_inference import Text2Text
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter


app = Flask(__name__)


class Token2Text(nn.Module):
    def __init__(
        self,
        mt_train_config: str,
        mt_model_file: str,
        bpemodel: str,
        token_list: str,
    ):
        super.__init__()
        self.mt_model = Text2Text(
            mt_train_config=mt_train_config,
            mt_model_file=mt_model_file,
            beam_size=20,
            ctc_weight=0.3,
            lm_weight=0.0,
        )
        self.tokenizer = build_tokenizer(
            token_type="bpe",
            bpemodel=bpemodel,
        )
        self.converter = TokenIDConverter(token_list=token_list)

    def inference(self, units: list[int]):
        # De-duplicate units
        deduplicated_units = [x[0] for x in groupby(units)]

        # units to cjk characters and apply BPE
        cjk_units = "".join([chr(int("4e00", 16) + c) for c in units])
        cjk_tokens = "".join([chr(int("4e00", 16) + c) for c in deduplicated_units])
        bpe_tokens = self.tokenizer.text2tokens(cjk_tokens)
        bpe_tokens = self.converter.tokens2ids(bpe_tokens)

        # Inference using the MT model
        bpe_tokens = torch.Tensor(bpe_tokens).to(self.mt_model.device)
        results = self.mt_model(bpe_tokens)

        return {
            "text": results[0][0],
            "units": cjk_units,
            "deduplicated_units": cjk_tokens,
        }


def load_model():
    return Token2Text()

device = "cpu"
model = load_model().to(device)

@app.route("/asr", methods=["POST"])
def asr():
    try:
        data = request.get_json()
        if not data or "tokens" not in data:
            return jsonify({"error": "Invalid input. Please provide a list of integers in \"tokens\"."}), 400

        numbers = data["tokens"]
        if not isinstance(numbers, list) or not all(isinstance(x, int) for x in numbers):
            return jsonify({"error": "Invalid input. \"numbers\" must be a list of integers."}), 400

        result = model(numbers)
        print(result)

        return jsonify({"text": result["text"]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=31589, debug=True)
