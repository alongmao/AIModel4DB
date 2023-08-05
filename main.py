import json

from flask import Flask, request

from CV.face_recognition import load_face_model, compare_faces
from NLP.sentiment_analysis import analyze_sentiment
from NLP.topic_recognition import extract_topic

app = Flask(__name__)
face_net, face_rec_model = load_face_model()


@app.route('/image/similarity')
def face_recognition():
    origin = request.args.get('origin')
    target = request.args.get('target')
    is_same_person, similarity_score = compare_faces(origin, target, face_net, face_rec_model)
    return {'similarity': similarity_score}


@app.route('/sentiment/predict', methods=['POST'])
def sentiment_predict():
    # 获取请求数据
    data = json.loads(request.get_data())
    text = data.get("text")
    sentiment_type, sentiment_score = analyze_sentiment(text)
    return {'sentiment_type': sentiment_type, 'sentiment_score': sentiment_score}


@app.route('/topic/extract', methods=['POST'])
def batch_extract_topic():
    # 获取请求数据
    data = json.loads(request.get_data())
    texts = list(data.get("texts"))

    response = extract_topic(texts)
    return {"data": response}


if __name__ == "__main__":
    app.run()
