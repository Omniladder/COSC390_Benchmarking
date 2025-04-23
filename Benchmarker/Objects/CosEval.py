from .BaseEval import BaseEval
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class CosEval(BaseEval):

    def __init__(self, embeddingModel: SentenceTransformer):
        self.embeddingModel = embeddingModel

    def evaluate(self, modelAnswer: str, trueAnswer: str) -> float:

        if(len(trueAnswer["text"]) > 0):
            answer = trueAnswer["text"][0]
        else:
            answer = ' '.join(trueAnswer["text"])
        
        
        modelEmbedding = self.embeddingModel.encode(modelAnswer)
        trueEmbedding = self.embeddingModel.encode(answer)
        return float(cosine_similarity([modelEmbedding], [trueEmbedding])[0][0])
