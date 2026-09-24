# File: app.py

import io
import base64
import os
import warnings
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from nlp_processor import NewsNLPProcessor
from wsd_module import WSDGRUClassifier

app = Flask(__name__, static_folder='frontend')
CORS(app)

print("[INFO] Initializing NLP Processing Engine...")
processor = NewsNLPProcessor()

print("[INFO] Pre-training GRU Word Sense Disambiguation Model...")
wsd_classifier = WSDGRUClassifier(target_word="bank")
wsd_definitions = wsd_classifier.train_model(epochs=50)
print("[INFO] Neural Network ready.")

@app.route('/')
def serve_frontend():
    return send_from_directory('frontend', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_article():
    data = request.json or {}
    article_text = data.get('article', '').strip()
    comparison_text = data.get('comparison_article', '').strip()

    if not article_text:
        return jsonify({"error": "Article text cannot be empty."}), 400

    if not comparison_text:
        comparison_text = "The Indian Central Bank adjusted interest rates today to reduce high inflation across markets."

    try:
        # 1. Basic Preprocessing
        clean_tokens = processor.preprocess_basic(article_text)

        # 2. Advanced Preprocessing
        no_stops, stemmed, lemmatized = processor.preprocess_advanced(clean_tokens)

        # 3. Morphological Analysis & Word Generation
        df_morph, word_gen = processor.morphological_analysis_and_generation(article_text[:300])
        morph_list = df_morph.to_dict(orient='records')

        # 4. N-Grams
        _, bigrams = processor.generate_ngrams(lemmatized, n=2)
        formatted_bigrams = [f"{bg[0][0]} {bg[0][1]} ({count})" for bg, count in bigrams]

        # 5. POS Tagging
        pos_tags = processor.perform_pos_tagging(no_stops[:20])

        # 6. Chunking & Feature Analysis
        chunks, feature_report = processor.perform_chunking_and_feature_analysis(pos_tags)

        # 7. Named Entity Recognition
        entities = processor.extract_entities(article_text)

        # 8. Document Similarity
        sim_score = float(processor.calculate_text_similarity(article_text, comparison_text))

        # 9. Word Sense Disambiguation (GRU)
        wsd_sense, wsd_conf = wsd_classifier.disambiguate(article_text)
        wsd_meaning = wsd_definitions[wsd_sense]

        # 10. Word Cloud (In-memory Base64 String)
        wc_file = processor.generate_word_cloud(article_text, "temp_wc.png")
        with open(wc_file, "rb") as image_file:
            encoded_wc = base64.b64encode(image_file.read()).decode('utf-8')

        # 11. Final News Article Summary
        summary = processor.summarize_article(article_text, num_sentences=2)

        return jsonify({
            "summary": summary,
            "wordcloud_base64": encoded_wc,
            "basic_tokens": clean_tokens[:15],
            "stopwords_count": len(clean_tokens) - len(no_stops),
            "lemmatized_sample": lemmatized[:12],
            "morphological_analysis": morph_list,
            "generated_words": [f"{item[0]} → {item[1]}" for item in word_gen],
            "bigrams": formatted_bigrams,
            "pos_tags": [f"{word} ({tag})" for word, tag in pos_tags[:10]],
            "chunks": [f"[{label}] {text}" for label, text in chunks],
            "feature_report": feature_report,
            "entities": [{"text": ent[0], "label": ent[1]} for ent in entities],
            "similarity_score": round(sim_score * 100, 2),
            "wsd_result": {
                "sense": wsd_meaning,
                "confidence": round(wsd_conf * 100, 2)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("[INFO] Starting server at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)