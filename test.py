"""
Creator & Developer : Muhammad Abrar
email : muhammadabrar9999@gmail.com
"""

from typing import List, Dict

import argparse
import signal
import sys
import csv
import json
import time
import string
import re
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"


import spacy
# from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class Paraphraser:

    def __init__(self, model: str, tokenizer: str):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model) 
        # self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer,model_max_length=128)
        self.spacy_trf_model = spacy.load('en_core_web_trf')

    @staticmethod
    def remove_punctuations_from_string(text: str):
        text_without_punctuations = str()
        # custom punctuation
        self_defined_punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”'
        if text is None:
            raise ValueError(f"Function parameter raw_text is {text}. Please correct your input.")
        for char in text:
            # if char not in string.punctuation:
            if char.lower() not in self_defined_punctuation.lower():
                text_without_punctuations += char
            else:
                text_without_punctuations += ' '
        text_without_punctuations = re.sub('\s+', ' ', text_without_punctuations).strip().lower()
        return text_without_punctuations

    def filter_out_similar_samples_from_list(self, samples: list, no_of_phrases:5) -> list:
        if not samples:
            raise ValueError("samples cannot be None.")
        if len(samples) <= no_of_phrases:
            return samples
        filtered_samples = list()
        filtered_samples_without_punctuations = list()
        for s in samples:
            filtered_s = self.remove_punctuations_from_string(s)
            if filtered_s not in filtered_samples_without_punctuations:
                filtered_samples_without_punctuations.append(filtered_s)
                filtered_samples.append(s)
        if len(filtered_samples) <= no_of_phrases:
            return samples
        return filtered_samples

    @staticmethod
    def cosine_filtration(corpus: list, sample: str, less_than: float, greater_than: float, *, num_seq=5) -> list:
        
        if len(corpus) <= num_seq:
            return corpus
        
        corpus = list(map(lambda corpus: corpus.lower(), corpus))
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(corpus)
        comparison_sample_tfidf_matrix = tfidf.transform([sample.lower()])
        cosine_sim = cosine_similarity(tfidf_matrix, comparison_sample_tfidf_matrix)
        flat_ndarray = cosine_sim.flatten()
        con_array = (flat_ndarray > greater_than) & (flat_ndarray < less_than)
        filtered_strings = [corpus[i] for i in range(len(con_array)) if con_array[i]]
        if len(filtered_strings) <= num_seq:
            return corpus
        return filtered_strings

    def sentence_splitter(self, raw_text: str) -> list:
        sentences = list()
        doc = self.spacy_trf_model(u'{}'.format(raw_text))
        for sent in doc.sents:
            # We should clean the text before splitting sentences
            filtered_sent = sent.text.replace('\n', ' ').replace('  ', '')
            filtered_sent = filtered_sent.strip()
            sentences.append(filtered_sent)
        return sentences

    def paraphrase_sentence(
            self,
            sentence_to_be_paraphrased,
            no_of_phrases=5,
            filter_with_punctuation=False,
            top_with_cosine=False,
            less_than=0.97,
            greater_than=0.3,

            # BART v.1 params
            num_beams=20,
            num_beam_groups=20,
            num_return_sequences=10,
            repetition_penalty=10.0,
            diversity_penalty=5.0,
            no_repeat_ngram_size=2,
            temperature=0.7,
            max_length=135,
            min_new_tokens=3
            # *, # After this we have to put 2 params: filteration and punctuations
            
    ) -> list:  
        
        # text =  "paraphrase: " + sentence_to_be_paraphrased + " </s>" 
        text = sentence_to_be_paraphrased
        encoding = self.tokenizer.encode_plus(
            text, return_tensors="pt", padding=True,
        )
        # input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        if len(input_ids[0]) > 128:
            max_length = len(input_ids[0])

        # import pdb;pdb.set_trace() # For Debugging
        outputs = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_masks,
            max_length=max_length,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            repetition_penalty=repetition_penalty,
            diversity_penalty=diversity_penalty,
            temperature=temperature,
            # penalty_alpha=penalty_alpha,
            # top_k=top_k,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            use_cache=True, # Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
            output_scores=True, # Whether or not to return the prediction scores. See scores under returned tensors for more details
            renormalize_logits=True, # search algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization. 
        )

        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
        if filter_with_punctuation:
            response = self.filter_out_similar_samples_from_list(samples=response,no_of_phrases=no_of_phrases)
        if top_with_cosine:
            response = self.cosine_filtration(corpus=response,sample=sentence_to_be_paraphrased,less_than=less_than,greater_than=greater_than,num_seq=no_of_phrases)

        # # Making sure to return response with specific no_of_phrases
        response = response[:no_of_phrases]

        # print("\n\n----------------")
        # print("Text: \n", sentence_to_be_paraphrased)
        # print("Query to model: \n", text)
        # print("Augments: \n")
        # for i, value in enumerate(response):
        #     print(f"{i}: {value}")

        return response

    def paraphrase(self, text: str, n=5) -> Dict[int,str]:
 
        # Removeing new-line characters
        if "\n" in text:
            text = text.replace("\n", " ")

        # Create a new dictionary for merged lists
        augment_merged_dict = dict()
        paraphrase = dict()
        
        text = text.strip()
        if text != "":
            sentences_list = self.sentence_splitter(raw_text=text)
            for index, sentence in enumerate(sentences_list):

                phrase_list = self.paraphrase_sentence(sentence_to_be_paraphrased=sentence, no_of_phrases=n, filter_with_punctuation=False,top_with_cosine=False)
                paraphrase[f"sent_{index}"] = phrase_list
        else: # handling the empty text, edge case
            ls = list()
            for i in range(n+1):
                ls.append(text)
            paraphrase[f"sent_0"] = ls

        # Iterate over the lists and merge elements at each index
        for index in range(len(paraphrase["sent_0"])):
            merged_list = [paraphrase[f'sent_{i}'][index] for i in range(len(paraphrase))]
            augment_merged_dict[index] = ' '.join(merged_list)

        return augment_merged_dict

def main(tokenizer_path, model_path):

    paraphraser_obj = Paraphraser(model_path, tokenizer_path)
    samples = [   
        "I hope you are in the best of your health.",
        "A paragraph is a series of sentences that are organized and coherent, and are all related to a single topic. Almost every piece of writing you do that is longer than a few sentences should be organized into paragraphs. This is because paragraphs show a reader where the subdivisions of an essay begin and end, and thus help the reader see the organization of the essay and grasp its main points.",
        "Please quote us the below to KSA including shipment charges.",
        "If you suspect any email as being a scam or fraudulent, please immediately delete the email. Do not respond, forward or click any links within the suspect email."
        ]
    
    try:
        for sample in samples:
            print("------------------------------------------")
            print("TEST SAMPLE: ", sample, "\n")
            paraphrase_json = dict()
            paraphrase_json["text"] = sample 
            paraphrase_json["paraphrases"] = dict()
            no_of_paraphrases = 20

            clones = paraphraser_obj.paraphrase(text=sample,n=no_of_paraphrases)
            for key, value  in clones.items():
                paraphrase_json["paraphrases"][f"{key}_clones"] = value
            
            json_formatted_str = json.dumps(paraphrase_json, indent=2)
            print(json_formatted_str)

    except KeyboardInterrupt:
        print("\n System Shuting Down.")
        sys.exit()

def start():

    checkpoint_path = "checkpoint-319912"
    model_path = checkpoint_path
    tokenizer_path = checkpoint_path
    main(tokenizer_path=tokenizer_path, model_path=model_path)


if __name__ == "__main__":
    start()
    
