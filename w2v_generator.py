import spacy
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


nlp = spacy.load("en_core_web_sm")



def train_word2vec(input_file, output_file, skipgram, loss, size, epochs):
    """
    train_word2vec(args**) -> Takes the input file,
    the output file and the model hyperparameters as
    arguments and trains the model accordingly.
    The model is saved at the output location.

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

    # for colab:
    #   vector_size   -> size
    #   epochs        -> iter
    model = Word2Vec(sentence, sg=skipgram, hs=loss,
                     vector_size=size, alpha=0.05, window=5,
                     min_count=1, workers=4, epochs=epochs)

    model.save(output_file)


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


# test train_word2vec
aircon = open('Aircon.sents')
train_word2vec(aircon, 'Aircon.w2v', 0, 0, 100, 2)
aircon_w2v = Word2Vec.load("Aircon.w2v")
aircon_kv = aircon_w2v.wv

print(aircon_kv.similarity('I', 'am'))
print(aircon_kv.vocab)

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