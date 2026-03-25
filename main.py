"""
Context RAG - PDF Quiz Generator
Streamlit uygulamasını başlatır.

Kullanım:
    streamlit run main.py
"""
import subprocess
import sys
import os


def main():
    app_path = os.path.join(os.path.dirname(__file__), "ui", "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


if __name__ == "__main__":
    main()
