import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

# --------- Config ----------
IMAGE_PATH = "VLM/log_images/traffic_intersection.avif"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------

def generate_caption(image):
    """
    Use BLIP image captioning to get a detailed natural language description.
    """
    print("Loading BLIP captioning model...")
    cap_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    cap_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    cap_model.eval()

    inputs = cap_processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = cap_model.generate(**inputs, max_new_tokens=50)

    caption = cap_processor.decode(out[0], skip_special_tokens=True)
    return caption

def answer_question(image, question: str):
    """
    Use BLIP VQA model to answer a natural language question about the image.
    """
    print(f"\n[Q] {question}")
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(DEVICE)
    vqa_model.eval()

    inputs = vqa_processor(images=image, text=question, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = vqa_model.generate(**inputs, max_new_tokens=20)

    answer = vqa_processor.decode(out[0], skip_special_tokens=True)
    print(f"[A] {answer}")
    return answer

def main():
    print(f"Using device: {DEVICE}")
    # 1) Load image
    image = Image.open(IMAGE_PATH).convert("RGB")

    # 2) Get a detailed caption
    caption = generate_caption(image)
    print("\n=== Generated Caption ===")
    print(caption)

    # 3) Ask some questions about the scene
    # You can modify / extend this list
    questions = [
        "How many cars are visible?",
        "Are there any pedestrians?",
        "Is this taken at an intersection?",
        "What time of day does it look like?",
        "Is the traffic heavy or light?",
        "Are there any traffic lights visible?",
    ]

    for q in questions:
        answer_question(image, q)

    # 4) Optional interactive loop
    print("\nYou can now ask your own questions about the image (type 'quit' to exit).")
    while True:
        q = input("\nYour question> ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        if not q:
            continue
        answer_question(image, q)

if __name__ == "__main__":
    main()
