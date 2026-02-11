# Semantic model module using Sentence Transformers - محاسبه شباهت معنایی

import numpy as np
from typing import List, Tuple
import hashlib
import json
import os
from dataclasses import dataclass
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

@dataclass
class SimilarityResult:
    text1: str
    text2: str
    similarity: float
    distance: float
    cached: bool = False

class SemanticModel:
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.cache = {}
        self.cache_file = os.path.join(config.cache_dir, "similarity_cache.json")
        
        # making the cache folder
        os.makedirs(config.cache_dir, exist_ok=True)
        
        self._load_cache()
        self._load_model()
    
    def _load_model(self):
        #Loading Sentence Transformers
        try:
            from sentence_transformers import SentenceTransformer, util
            
            logger.info(f"loading model {self.config.model_name}...")
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
            self.util = util
            logger.info("model loaded successfully")
            
        except ImportError as e:
            logger.error(f"could not load model: {e}")
            raise
    
    def _load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"cache loaded {len(self.cache)}")
        except Exception as e:
            logger.warning(f"could not load cache {e}")
            self.cache = {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"failed to save cache: {e}")
    
    def _get_cache_key(self, text1: str, text2: str) -> str:
        sorted_texts = tuple(sorted([text1, text2]))
        return hashlib.md5(json.dumps(sorted_texts, ensure_ascii=False).encode()).hexdigest()
    
    def compute_similarity(self, text1: str, text2: str) -> SimilarityResult:
        cache_key = self._get_cache_key(text1, text2)
        
        if self.config.enable_cache and cache_key in self.cache:
            similarity = self.cache[cache_key]
            cached = True
        else:
            similarity = self._compute_similarity_impl(text1, text2)
            cached = False
            
            if self.config.enable_cache:
                self.cache[cache_key] = similarity
                if len(self.cache) % 100 == 0:
                    self._save_cache()
        
        # semantic distance
        distance = 1 - similarity
        
        return SimilarityResult(
            text1=text1,
            text2=text2,
            similarity=similarity,
            distance=distance,
            cached=cached
        )
    
    def _compute_similarity_impl(self, text1: str, text2: str) -> float:

        try:
            embeddings = self.model.encode(
                [text1, text2],
                batch_size=self.config.batch_size,
                convert_to_tensor=True,
                device=self.config.device
            )
            
            cosine_score = self.util.cos_sim(embeddings[0], embeddings[1])
            similarity = float(cosine_score[0][0])
            
            # turning the numbers to fit in [0,1]
            similarity = (similarity + 1) / 2
            similarity = max(0.0, min(1.0, similarity))
            
            return similarity
            
        except Exception as e:
            logger.error(f"failed to compute similarity: {e}")
            return 0
    
    def compute_batch_similarities(self, text_pairs: List[Tuple[str, str]]) -> List[SimilarityResult]:
        # computing in groups
        results = []
        
        for text1, text2 in tqdm(text_pairs, desc="computing similarities"):
            result = self.compute_similarity(text1, text2)
            results.append(result)
        
        return results
    
    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        matrix = np.eye(n)
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_tensor=True,
            device=self.config.device
        )
        
        similarity_matrix = self.util.cos_sim(embeddings, embeddings)

        # converting to numpy and normalizing
        similarity_matrix = similarity_matrix.cpu().numpy()
        similarity_matrix = (similarity_matrix + 1) / 2
        similarity_matrix = np.clip(similarity_matrix, 0, 1)
        
        return similarity_matrix
    
    def save_model(self, path: str):
        if self.model:
            self.model.save(path)
            logger.info(f"model saved in this path: {path}")
    
    def load_model(self, path: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(path)
        logger.info(f"model loaded from: {path}")