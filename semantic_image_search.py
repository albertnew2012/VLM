import os
import torch
import clip
from PIL import Image

# --------- Config ----------
IMAGE_DIR = "log_images"   # folder under current working dir
TOP_K = 5                  # how many results to show
# ---------------------------

def load_image_paths(image_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".avif", ".webp"}
    paths = []
    for fname in os.listdir(image_dir):
        if os.path.splitext(fname.lower())[1] in exts:
            paths.append(os.path.join(image_dir, fname))
    paths.sort()
    return paths

def build_image_index(model, preprocess, image_paths, device):
    """
    Compute CLIP features for all images and return:
      - image_features: [N, 512] normalized tensor
      - image_paths: list of paths in same order
    """
    all_feats = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                img = preprocess(img)
                images.append(img)
            if not images:
                continue

            images = torch.stack(images, dim=0).to(device)  # [B, 3, 224, 224]
            feats = model.encode_image(images)               # [B, 512]
            # Normalize for cosine similarity
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())  # keep on CPU to save GPU mem

    if not all_feats:
        raise RuntimeError("No images found to index.")
    image_features = torch.cat(all_feats, dim=0)  # [N, 512]
    return image_features, image_paths

def search_images(
    model,
    image_features,
    image_paths,
    device,
    query,
    top_k=5,
    min_score=None,   # e.g., 0.2 or 0.25
):
    """
    Given a text query, compute similarity to all image features
    and return top_k matches with optional similarity threshold.
    """
    with torch.no_grad():
        text_tokens = clip.tokenize([query]).to(device)      # [1, 77]
        text_feat = model.encode_text(text_tokens)           # [1, 512]
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        img_feats = image_features.to(device)                # [N, 512]
        sims = text_feat @ img_feats.T                       # [1, N]
        sims = sims.squeeze(0)                               # [N]

        # Optional threshold filter
        if min_score is not None:
            mask = sims >= min_score
            if not mask.any():
                return []  # no image passes the threshold
            sims_filtered = sims[mask]
            paths_filtered = [p for p, keep in zip(image_paths, mask.tolist()) if keep]
        else:
            sims_filtered = sims
            paths_filtered = image_paths

        # Now do top-k on the filtered sims
        k = min(top_k, sims_filtered.numel())
        values, indices = sims_filtered.topk(k)

    values = values.cpu()
    indices = indices.cpu()

    results = []
    for score, idx in zip(values, indices):
        results.append((paths_filtered[idx], float(score.item())))
    return results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # 1) Load image paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, IMAGE_DIR)
    print(f"Indexing images from: {image_dir}")

    image_paths = load_image_paths(image_dir)
    if not image_paths:
        print("No images found. Check IMAGE_DIR and image formats.")
        return

    # 2) Build index (compute CLIP features for all images)
    image_features, image_paths = build_image_index(
        model, preprocess, image_paths, device
    )
    print(f"Indexed {len(image_paths)} images.")

    # 3) Interactive search loop
    print("\nType a natural language query (or 'quit' to exit).")
    while True:
        query = input("\nQuery> ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            break
        if not query:
            continue

        results = search_images(
            model, image_features, image_paths, device, query, top_k=TOP_K, min_score=0.22, 
        )

        print(f"\nTop {len(results)} matches for: \"{query}\"")
        for path, score in results:
            print(f"  score={score:.4f}  file={path}")

if __name__ == "__main__":
    main()
