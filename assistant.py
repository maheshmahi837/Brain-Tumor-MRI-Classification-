"""
assistant.py

Lightweight Project-Aware RAG Assistant

Features
--------
✓ Automatic project indexing
✓ Reads Python source files
✓ Reads README / TXT / JSON
✓ Reads report.pdf
✓ Uses Gemini 2.5 Flash
✓ Retrieves only relevant project files
✓ Can analyze evaluation figures
✓ No vector database
✓ No LangChain
✓ No FAISS
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict

import fitz                      # PyMuPDF
from PIL import Image

from google import genai
from google.genai import types


################################################################################
# CONFIGURATION
################################################################################

PROJECT_ROOT = Path(os.getcwd())

SUPPORTED_TEXT = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".tex",
    ".yaml",
    ".yml"
}

SUPPORTED_IMAGES = {
    ".png",
    ".jpg",
    ".jpeg"
}

SYSTEM_PROMPT = """
You are an AI assistant for this Brain Tumor MRI Classification project.

Answer ONLY using the supplied project files and live context data.

Never invent implementation details. Rely exactly on the text and images provided.

If the information is unavailable, say that the project does not contain it.

This project is a research prototype and must never be interpreted as a clinical tool.
"""


################################################################################
# FILE INDEXER
################################################################################

class ProjectIndexer:

    def __init__(self, root: Path = PROJECT_ROOT):
        self.root = root
        self.text_files = {}
        self.image_files = {}
        self.report_text = ""
        self.build_index()

    def build_index(self):
        for path in self.root.rglob("*"):
            if path.is_dir() or "voxelgrids_env" in path.parts or ".git" in path.parts:
                continue

            suffix = path.suffix.lower()

            if suffix in SUPPORTED_TEXT:
                self.text_files[path.name] = path
            elif suffix in SUPPORTED_IMAGES:
                self.image_files[path.name] = path
            elif path.name.lower() == "report.pdf":
                self.report_text = self.read_pdf(path)
                self.text_files["report.pdf"] = path

    @staticmethod
    def read_pdf(path: Path):
        try:
            doc = fitz.open(path)
            text = []
            for page in doc:
                text.append(page.get_text())
            doc.close()
            return "\n".join(text)
        except Exception:
            return ""

    def read_text_file(self, path: Path):
        try:
            if path.name == "report.pdf":
                return self.report_text
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

################################################################################
# RETRIEVER
################################################################################

class Retriever:

    def __init__(self, indexer: ProjectIndexer):
        self.indexer = indexer

    @staticmethod
    def tokenize(text):
        return set(re.findall(r"[a-zA-Z_]+", text.lower()))

    def score_file(self, question, filename):
        q = self.tokenize(question)
        f = self.tokenize(filename)
        return len(q & f)

    def retrieve_text_files(self, question, top_k=5):
        scores = []
        for name, path in self.indexer.text_files.items():
            score = self.score_file(question, name)
            text = self.indexer.read_text_file(path)
            if text:
                score += len(self.tokenize(question) & self.tokenize(text[:5000]))
            scores.append((score, name, path))

        scores.sort(reverse=True)
        selected = []
        for score, name, path in scores:
            if score > 0 or name in ["README.md", "report.pdf", "metrics_summary.json"]:
                selected.append(path)
            if len(selected) >= top_k:
                break
        return selected

################################################################################
# IMAGE RETRIEVER
################################################################################

class ImageRetriever:

    def __init__(self, indexer: ProjectIndexer):
        self.indexer = indexer

    def retrieve(self, question: str):
        q = question.lower()
        images = []
        mapping = {
            "confusion": ["confusion_matrix.png"],
            "matrix": ["confusion_matrix.png"],
            "roc": ["roc_curves.png"],
            "auc": ["roc_curves.png"],
            "curve": ["training_curves.png", "roc_curves.png"],
            "training": ["training_curves.png"],
            "loss": ["training_curves.png"],
            "accuracy": ["training_curves.png"],
            "prediction": ["sample_predictions.png"],
            "sample": ["sample_predictions.png"],
            "failure": ["failure_cases.png"],
            "error": ["failure_cases.png"],
            "misclassified": ["failure_cases.png"]
        }

        for keyword, files in mapping.items():
            if keyword in q:
                for f in files:
                    if f in self.indexer.image_files:
                        images.append(self.indexer.image_files[f])

        return list(dict.fromkeys(images))

################################################################################
# CONTEXT BUILDER
################################################################################

class ContextBuilder:

    def __init__(self, indexer, retriever, image_retriever):
        self.indexer = indexer
        self.retriever = retriever
        self.image_retriever = image_retriever

    def build(self, question):
        files = self.retriever.retrieve_text_files(question, top_k=5)
        images = self.image_retriever.retrieve(question)
        context = []
        used_files = []

        for file in files:
            text = self.indexer.read_text_file(file)
            if not text.strip():
                continue
            used_files.append(file.name)
            context.append(f"\n=========================\nFILE : {file.name}\n=========================\n\n{text[:8000]}\n")

        return "\n".join(context), images, used_files

################################################################################
# GEMINI RAG INTERFACE (Drop-in Replacement for ProjectAssistant)
################################################################################

class ProjectAssistant:

    def __init__(self, model="gemini-2.5-flash"):
        self.client = genai.Client()
        self.model = model
        self.indexer = ProjectIndexer()
        self.retriever = Retriever(self.indexer)
        self.image_retriever = ImageRetriever(self.indexer)
        self.context_builder = ContextBuilder(self.indexer, self.retriever, self.image_retriever)
        self.history = []

    def reset(self):
        self.history = []

    def ask(self, question: str) -> str:
        # Dynamically build context based on the incoming question
        context_text, images, used_files = self.context_builder.build(question)
        
        # Build prompt with instruction, context, and history reference
        prompt = f"""
SYSTEM
{SYSTEM_PROMPT}

PROJECT CONTEXT FILES INJECTED: {', '.join(used_files)}
{context_text}

USER QUESTION:
{question}
"""
        # Prepare contents array for the multimodal API call
        contents = []
        
        # Add text prompt component
        contents.append(types.Part.from_text(text=prompt))
        
        # Add retrieved images as standard image parts
        for img_path in images:
            try:
                img = Image.open(img_path)
                contents.append(img)
            except Exception:
                pass

        # Call Gemini using the official SDK interface
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents
        )
        return response.text


def chat_loop():
    """Interactive CLI chat loop."""
    assistant = ProjectAssistant()
    print("\n🧠 Brain Tumor MRI — Advanced RAG Project Assistant (Powered by Gemini)")
    print("    Type 'quit' to exit | 'reset' to clear history\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "reset":
            assistant.reset()
            print("[History cleared]\n")
            continue

        answer = assistant.ask(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    chat_loop()