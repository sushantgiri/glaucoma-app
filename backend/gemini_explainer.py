import os
from typing import Dict, List, Optional

import google.generativeai as genai


class GeminiExplainer:
    """
    Wraps Google's Generative AI API and automatically picks a model
    that supports generateContent.
    """

    def __init__(self, api_key_env: str = "GOOGLE_API_KEY"):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} not set. Please export your Gemini API key "
                "or add it to Streamlit Secrets as GOOGLE_API_KEY."
            )

        genai.configure(api_key=api_key)

        self.model_name = self._select_model_name()
        self.model = genai.GenerativeModel(self.model_name)
        print(f"[GeminiExplainer] Using model: {self.model_name}")

    def _select_model_name(self) -> str:
        models = genai.list_models()
        text_models: List[str] = []

        for m in models:
            name = getattr(m, "name", "")
            methods = getattr(m, "supported_generation_methods", [])
            if "generateContent" in methods:
                text_models.append(name)

        if not text_models:
            raise RuntimeError(
                "No Gemini models in this project support generateContent. "
                "Check your API key/project in Google AI Studio."
            )

        preferred = [n for n in text_models if "gemini" in n.lower()]
        return preferred[0] if preferred else text_models[0]

    def build_prompt(
        self,
        result: Dict,
        gradcam_path: Optional[str] = None,
    ) -> str:
        pred_label = result["pred_label"]
        p0, p1 = result["probs"]

        prompt = f"""
You are an ophthalmologist AI assistant.

A DenseNet121 glaucoma detection model produced the following result for a single retinal fundus image:

Prediction source: DenseNet121
Prediction: {pred_label}
Probabilities: Normal = {p0:.3f}, Glaucoma = {p1:.3f}

Explainability:
- Grad-CAM heatmap available: {"yes" if gradcam_path else "no"}.

Please write a concise ophthalmologist-style explanation (5–8 sentences) that:
1) Describes the level of suspicion for glaucoma based on the probabilities.
2) If Grad-CAM is available, briefly mention what it suggests qualitatively (optic disc / RNFL regions) without overclaiming.
3) Clearly state that this is a research tool and not a clinical diagnosis.

Use plain English suitable for a clinician reading a research prototype output.
"""
        return prompt.strip()

    def explain(
        self,
        result: Dict,
        gradcam_path: Optional[str] = None,
    ) -> str:
        prompt = self.build_prompt(result, gradcam_path)
        response = self.model.generate_content(prompt)
        return response.text.strip()
