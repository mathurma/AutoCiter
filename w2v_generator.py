import spacy
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence


nlp = spacy.load("en_core_web_sm")


def get_keyedvec(input_file, output_file, skipgram, loss, size, epochs):
    """
    get_keyedvec(args**) -> Takes the input file,
    the output file and the model hyperparameters as
    arguments and trains a word2vec model accordingly.
    The model's Keyed Vectors are saved at the output location.

    Arguments
    ---------
    input_file : Input pre-processed text dump
    output_file : Output directory to save the model.
    skipgram : Layers of the model (0 - CBOW, 1 - Skipgram)
    loss : Loss Function (0 - Negative Sampling, 1 - Heirarichal Loss)
    size : Embedding size (100 ~ 300)
    epochs : Number of epochs
    """
    sentence = LineSentence(input_file)

    model = Word2Vec(sentence, sg=skipgram, hs=loss,
                     size=size, alpha=0.05, window=5,
                     min_count=1, workers=4, iter=epochs)
    keyedvec = model.wv

    keyedvec.save(output_file)


def get_sents(text: str, nlp_model):
    sentences = []
    doc = nlp_model(text)
    for sentence in doc.sents:
        sentences.append(" ".join(sentence.text.split()))
    return sentences


def lines_to_file(output_file: str, sentences: list):
    f = open(output_file, "w")
    f.writelines("%s\n" % s for s in sentences)
    f.close()


# test get_keyedvec
aircon = open('Aircon.sents')
get_keyedvec(aircon, 'Aircon.kv', 0, 0, 100, 2)
aircon_kv = KeyedVectors.load("Aircon.kv")

print(aircon_kv.similarity('I', 'am'))

# test get_sents
text = """                               AIR CONDITIONING



              My telephone receiver slams down on its cradle.  I'm
          upset. I am soaked to the skin, sweat runs from my brow.
          The air conditioner that I so naively  entrusted to the
          Yellow Pages Repair shop is delayed another two weeks."""
sents = get_sents(text, nlp)
print(sents)

# test lines_to_file
lines_to_file('Aircon.sents', sents)
f = open("Aircon.sents", "r")
print(f.read())