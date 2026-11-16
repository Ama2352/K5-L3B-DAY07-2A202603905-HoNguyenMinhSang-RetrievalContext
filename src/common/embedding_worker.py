"""Real Day 6 BGE embeddings in a short-lived worker to release ONNX memory."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


class ProcessEmbedder:
    model_name='BAAI/bge-small-en-v1.5'
    dimensions=384

    def __init__(self,day6):
        self.day6=Path(day6).resolve()

    def _call(self,payload):
        result=subprocess.run([sys.executable,'-m','src.common.embedding_worker','--day6',str(self.day6)],
            input=json.dumps(payload),capture_output=True,text=True,encoding='utf8',timeout=180)
        if result.returncode:
            # Never echo subprocess stderr: dependency errors may contain private environment/config values.
            raise RuntimeError(f'Embedding worker failed (exit {result.returncode}); no mock vector substituted')
        vectors=json.loads(result.stdout)
        if any(len(v)!=self.dimensions for v in vectors):raise ValueError('Embedding dimension mismatch')
        return vectors

    def embed_query(self,question):
        return self._call({'query':question})[0]

    def embed_documents(self,texts):
        return self._call({'documents':texts})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day6',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.day6.resolve()))
    from starter.src.embeddings import MyEmbedder
    embedder=MyEmbedder(cache_dir=str(args.day6.resolve()/'.cache/models'))
    payload=json.load(sys.stdin)
    vectors=([embedder.embed_query(payload['query'])] if 'query' in payload
             else embedder.embed_documents(payload['documents']))
    print(json.dumps(vectors))


if __name__=='__main__':main()
