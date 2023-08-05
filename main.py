from flask import Flask, request

from NLP.face_recognition import load_face_model, compare_faces

app = Flask(__name__)
face_net, face_rec_model = load_face_model()

@app.route('/face/recognition')
def face_recognition():
    origin = request.args.get('origin')
    target = request.args.get('target')
    is_same_person, similarity_score = compare_faces(origin, target, face_net, face_rec_model)
    return {'similarity': similarity_score}


if __name__ == "__main__":
    app.run()
