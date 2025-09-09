import subprocess
import sys

def setup_environment():
    print("Setting up the RAG system environment...")
    
    # Install requirements
    print("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install spaCy model
    print("Installing spaCy language model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    print("\nSetup completed successfully!")
    print("Make sure to configure your .env file with the necessary credentials.")

if __name__ == "__main__":
    setup_environment()