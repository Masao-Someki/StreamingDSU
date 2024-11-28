from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/asr", methods=["POST"])
def process_list():
    try:
        # Parse JSON input
        data = request.get_json()
        if not data or "tokens" not in data:
            return jsonify({"error": "Invalid input. Please provide a list of integers in \"tokens\"."}), 400

        numbers = data["tokens"]
        if not isinstance(numbers, list) or not all(isinstance(x, int) for x in numbers):
            return jsonify({"error": "Invalid input. \"numbers\" must be a list of integers."}), 400

        # Process the list of integers
        result = f"You sent {len(numbers)} integers. Their sum is {sum(numbers)}."

        # Return the result as a string
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=41589, debug=True)