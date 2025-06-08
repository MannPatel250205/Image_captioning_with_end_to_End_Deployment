from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from src.Model.pipeline.predict import PredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)



@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("dvc repro")
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if not os.path.exists("artifacts"):
            print("Artifacts folder not found. Running training first...")
            os.system("python main.py")
            print("Training completed.")
        file = request.files.get("image")

        upload_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)

        # Save the uploaded file
        file.save(upload_path)

        if not os.path.exists(upload_path):
            print("File was not saved correctly")
        

        # Run prediction
        prediction = PredictionPipeline(upload_path)
        captions = prediction.predict()
        return jsonify({"caption": captions})
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Failed to generate caption"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080) #local host
    # app.run(host='0.0.0.0', port=8080) #for AWS
    # app.run(host='0.0.0.0', port=80) #for AZURE
