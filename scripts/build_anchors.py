import argparse
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from tqdm import tqdm

def build_anchors(labels_file, out_path, model_name='sentence-transformers/all-MiniLM-L6-v2', batch_size=32):
    labels = [l.strip() for l in open(labels_file).read().splitlines() if l.strip()]
    model = SentenceTransformer(model_name)
    anchors = {}
    for lbl in tqdm(labels, desc="Building anchors"):
        prompt = f"This is an image of {lbl}"
        vec = model.encode(prompt)  # numpy array
        anchors[lbl] = vec
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(anchors, f)
    print(f"Saved anchors to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-file', type=str, required=True)
    parser.add_argument('--out', type=str, default='anchors/anchors.pkl')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    args = parser.parse_args()
    build_anchors(args.labels_file, args.out, args.model)
