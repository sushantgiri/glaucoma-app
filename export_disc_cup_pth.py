import torch
from transformers import SegformerForSemanticSegmentation

MODEL_ID = "pamixsun/segformer_for_optic_disc_cup_segmentation"
OUT_PATH = "disc_cup_segformer_best.pth"

model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
model.eval()

# Save ONLY weights (state_dict) as a .pth file
torch.save(model.state_dict(), OUT_PATH)

print("Saved:", OUT_PATH)
