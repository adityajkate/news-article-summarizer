# File: main.py

import os
import warnings

# Suppress warnings and environment noise before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from nlp_processor import NewsNLPProcessor
from wsd_module import WSDGRUClassifier

def main():
    print("===================================================================")
    print(" REQUIREMENT 1: REAL-WORLD PROJECT FORMULATION & STATEMENT")
    print("===================================================================")
    print("Project Title: News Article Summarizer & Linguistic Analyzer")
    print("Application Domain: Automated News Processing and Information Extraction")
    print("Problem Statement: Digital news streams contain large volumes of unstructured")
    print("data. This system extracts key summaries, identifies entities, runs contextual")
    print("word sense disambiguation via GRU, and generates syntactic analysis.\n")

    article_text = (
        "The Central Bank of India announced a historic monetary policy adjustment on Wednesday. "
        "The governing bank aims to curb surging inflation by steadily raising interest rates. "
        "Financial market analysts and institutional investors expressed surprise over the timing. "
        "Meanwhile, torrential rains caused the heavy river bank to overflow in northern provinces, "
        "causing severe flooding across local communities. The national government is delivering emergency "
        "relief supplies to support families residing near the river bank."
    )

    processor = NewsNLPProcessor()

    # Requirement 11: Word Cloud Generation
    wc_file = processor.generate_word_cloud(article_text)
    print(f"[REQ 11] Word Cloud generated successfully: Saved to '{wc_file}'")

    # Requirement 2: Basic Preprocessing
    clean_tokens = processor.preprocess_basic(article_text)
    print(f"\n[REQ 2] Basic Tokens (Filtered & Script Validated): {clean_tokens[:8]}")

    # Requirement 3: Advanced Preprocessing
    no_stops, stemmed, lemmatized = processor.preprocess_advanced(clean_tokens)
    print(f"[REQ 3] Stopwords Removed Count: {len(clean_tokens) - len(no_stops)}")
    print(f"[REQ 3] Lemmatized Sample: {lemmatized[:8]}")

    # Requirement 4: Morphological Analysis & Word Generation
    df_morph, word_gen = processor.morphological_analysis_and_generation("Central Bank increased interest rates rapidly.")
    print("\n[REQ 4] Morphological Analysis:")
    print(df_morph.to_string(index=False))
    print(f"[REQ 4] Word Generation Examples: {word_gen}")

    # Requirement 5: N-Gram Analysis
    _, bigram_freqs = processor.generate_ngrams(lemmatized, n=2)
    print(f"\n[REQ 5] Top Bigrams: {bigram_freqs}")

    # Requirement 6: POS Tagging
    pos_tags = processor.perform_pos_tagging(no_stops)
    print(f"\n[REQ 6] POS Tags (First 6): {pos_tags[:6]}")

    # Requirement 7: Chunking & Feature Analysis
    chunks, feature_report = processor.perform_chunking_and_feature_analysis(pos_tags)
    print(f"\n[REQ 7] Extracted Chunks: {chunks}")
    print(f"[REQ 7] Feature Selection Analysis: {feature_report['Feature 2']} | {feature_report['Training Size Impact']}")

    # Requirement 8: Named Entity Recognition
    entities = processor.extract_entities(article_text)
    print(f"\n[REQ 8] Named Entities Recognized: {entities}")

    # Requirement 9: Text Similarity
    comparison_article = "The Indian Central Bank adjusted interest rates today to reduce high inflation across markets."
    sim_score = processor.calculate_text_similarity(article_text, comparison_article)
    print(f"\n[REQ 9] Document Cosine Similarity Score: {sim_score:.4f}")

    # Requirement 10: Word Sense Disambiguation using GRU
    print("\n[REQ 10] Training GRU Neural Network for Word Sense Disambiguation (WSD)...")
    wsd_classifier = WSDGRUClassifier(target_word="bank")
    definitions = wsd_classifier.train_model(epochs=60)
    
    context_1 = "The governing central bank raised interest rates to control economic inflation."
    context_2 = "Water overflowed rapidly along the muddy river bank during the rainstorm."

    sense_1, conf_1 = wsd_classifier.disambiguate(context_1)
    sense_2, conf_2 = wsd_classifier.disambiguate(context_2)

    print(f" Context 1: '{context_1}'")
    print(f"  -> Predicted Sense: [{definitions[sense_1]}] (Confidence: {conf_1:.2%})")
    print(f" Context 2: '{context_2}'")
    print(f"  -> Predicted Sense: [{definitions[sense_2]}] (Confidence: {conf_2:.2%})")

    # Project Final Output: News Summary
    print("\n===================================================================")
    print(" PROJECT DELIVERABLE: GENERATED NEWS ARTICLE SUMMARY")
    print("===================================================================")
    summary = processor.summarize_article(article_text, num_sentences=2)
    print(f"{summary}\n")

if __name__ == "__main__":
    main()