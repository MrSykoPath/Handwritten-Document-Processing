import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import pipeline  # import your original code

# Load env variables
load_dotenv()

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Flask backend is running 🎉"})

# Route to trigger your pipeline
@app.route("/run-pipeline", methods=["POST"])
def run_pipeline():
    try:
        data = request.get_json()

        # Extract inputs
        source_folder_id = data.get("source_folder_id")
        result_folder_id = data.get("result_folder_id")
        ai_model_name = data.get("ai_model_name")
        api_key = data.get("api_key")

        # Check missing inputs
        if not source_folder_id:
            return jsonify({"error": "Source folder ID is required"}), 400
        if not result_folder_id:
            return jsonify({"error": "Result folder ID is required"}), 400
        if not ai_model_name:
            return jsonify({"error": "AI model name is required"}), 400
        if not api_key:
            return jsonify({"error": "API key is required"}), 400

        # Call your pipeline with error handling
        try:
            pipeline.process_drive_folder(
                source_folder_id, result_folder_id, ai_model_name, api_key
            )
        except FileNotFoundError as e:
            return jsonify({"error": f"File not found: {str(e)}"}), 404
        except PermissionError:
            return jsonify({"error": "Permission denied. Check your Google Drive access rights"}), 403
        except ValueError as e:
            return jsonify({"error": f"Invalid input: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": f"Unexpected pipeline error: {str(e)}"}), 500

        return jsonify({"status": "Pipeline completed successfully ✅"})

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
