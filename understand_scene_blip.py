import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

IMAGE_PATH = "log_images/traffic_intersection.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print("Loading BLIP models once...")

cap_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
cap_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(DEVICE).eval()

vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-capfilt-large"
).to(DEVICE).eval()


def generate_caption(image):
    inputs = cap_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = cap_model.generate(**inputs, max_new_tokens=50)
    return cap_processor.decode(out[0], skip_special_tokens=True)


def answer_question(image, question: str):
    print(f"\n[Q] {question}")
    inputs = vqa_processor(images=image, text=question, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = vqa_model.generate(**inputs, max_new_tokens=20)
    answer = vqa_processor.decode(out[0], skip_special_tokens=True)
    print(f"[A] {answer}")
    return answer


def main():
    image = Image.open(IMAGE_PATH).convert("RGB")

    caption = generate_caption(image)
    print("\n=== Generated Caption ===")
    print(caption)

    questions = [
        "How many cars are visible?",
        "How many pedestrians are visible?",
        "Is this taken at an intersection?",
        "What time of day does it look like?",
        "Is the traffic heavy or light?",
        "Are there any traffic lights visible?",
    ]
    for q in questions:
        answer_question(image, q)

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
