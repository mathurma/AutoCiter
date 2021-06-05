import spacy
import gensim
import smart_open
from gensim.models import Doc2Vec
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
    sentences = LineSentence(open(input_file))
    model = Word2Vec(sentences, sg=skipgram, hs=loss,
                     vector_size=size, alpha=0.05, window=5,
                     min_count=1, workers=4, epochs=epochs)

    keyedvec = model.wv
    keyedvec.save(output_file)


def get_doc2vec(input_file, output_file, skipgram, loss, size, epochs):
    paragraphs = list(read_corpus(input_file))
    model = Doc2Vec(paragraphs, vector_size=size, min_count=1, epochs=epochs)

    model.save(output_file)


def read_corpus(fname, tokens_only=False):
  # The file we’re reading is a corpus. Each line of the file is a document (paragraph).
  with smart_open.open(fname, encoding="iso-8859-1") as f:
    for i, line in enumerate(f):
      tokens = gensim.utils.simple_preprocess(line) # lowercases, tokenizes
      if tokens_only:
        yield tokens
      else:
        # For training data, add tags
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

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


def get_similar(model, paragraph, num_sim):
  tokens = gensim.utils.simple_preprocess(paragraph)
  inferred_vector = model.infer_vector(tokens)
  similar = model.docvecs.most_similar([inferred_vector], topn=num_sim) #get the most similar
  return similar

# test get_keyedvec
train_file = 'Aircon.sents'

get_keyedvec(train_file, 'Aircon.kv', 0, 0, 100, 2) # lower all the text
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

# test get_doc2vec
train_file = 'Aircon.txt'

get_doc2vec(train_file, 'Aircon.d2v', 0, 0, 50, 40)
aircon_d2v = Doc2Vec.load("Aircon.d2v")

similar = get_similar(aircon_d2v, "I was shaking my head", 1)
print(similar)