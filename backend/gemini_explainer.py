import os
from typing import Dict, List

import google.generativeai as genai


class GeminiExplainer:
    """
    Wraps Google's Generative AI API and automatically picks a model
    that supports generateContent, based on what is available in
    your project. This avoids 404 errors for hard-coded model names.
    """

    def __init__(self, api_key_env: str = "GOOGLE_API_KEY"):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} not set. Please export your Gemini API key "
                "or add it to Streamlit Secrets as GOOGLE_API_KEY."
            )

        # Configure Gemini client
        genai.configure(api_key=api_key)

        # Dynamically pick a model that supports generateContent
        self.model_name = self._select_model_name()
        self.model = genai.GenerativeModel(self.model_name)
        print(f"[GeminiExplainer] Using model: {self.model_name}")

    def _select_model_name(self) -> str:
        """
        Use genai.list_models() to find a model that supports
        generateContent. Prefer gemini-based models if available.
        """
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
        if preferred:
            return preferred[0]

        return text_models[0]

    def build_prompt(
        self,
        ensemble: Dict,
        per_model: Dict,
        gradcam_paths: Dict[str, str],
        vit_rollout_path: str = "",
    ) -> str:
        pred_label = ensemble["pred_label"]
        probs = ensemble["probs"]
        normal_prob = probs[0]
        glaucoma_prob = probs[1]

        per_model_lines: List[str] = []
        for name, info in per_model.items():
            p0, p1 = info["probs"]
            per_model_lines.append(
                f"- {name}: {info['pred_label']} "
                f"(Normal: {p0:.3f}, Glaucoma: {p1:.3f})"
            )

        gradcam_models = ", ".join(gradcam_paths.keys()) if gradcam_paths else "none"

        prompt = f"""
You are an ophthalmologist AI assistant.

A glaucoma detection ensemble model produced the following result for a single retinal fundus image:

Ensemble prediction: {pred_label}
Ensemble probabilities: Normal = {normal_prob:.3f}, Glaucoma = {glaucoma_prob:.3f}

Per-model predictions:
{chr(10).join(per_model_lines)}

Explainability information:
- Grad-CAM heatmaps are available for: {gradcam_models}.
- ViT attention rollout image path: {vit_rollout_path or "not generated"}.

Please write a concise ophthalmologist-style explanation (5–8 sentences) that:
1. Describes the level of suspicion for glaucoma based on the probabilities.
2. Mentions how consistent the models are with each other.
3. Qualitatively explains what the heatmaps suggest about optic disc and nerve fiber layer.
4. Clearly states that this is a research tool and not a clinical diagnosis.

Use plain English suitable for a clinician reading a research prototype output.
"""
        return prompt

    def explain(
        self,
        ensemble: Dict,
        per_model: Dict,
        gradcam_paths: Dict[str, str],
        vit_rollout_path: str = "",
    ) -> str:
        prompt = self.build_prompt(ensemble, per_model, gradcam_paths, vit_rollout_path)
        response = self.model.generate_content(prompt)
        return response.text.strip()
