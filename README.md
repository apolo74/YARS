# YARS
Yet Another RAG Script

The main goal of this project is to share a light-weight but powerful and open-source base for people to ingest PDF files and find information directly by asking questions. The core components of this chatbot are **nvidia** and **langchain** libraries, both **free** to use! The main advantage of this implementation is the possibility to choose from a list of different **LLM** models hosted by NVIDIA.

**YARS** will continue to grow with more functionality, with the target audience being the scientific community. I'll write more about my personal short- and long-term goals of this *chatbot* soon :)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

**YARS** is developed with **Python**, so we start there and it is assumed that you have a working version of Python in your system. If not, then I recommend to follow the instructions from the [Python](https://www.python.org/) website. There are also thousands of tutorials on the web, one I recommend is [The Hitchhiker's Guide to Python](https://docs.python-guide.org/starting/installation/#installation). Go for **Python 3.8** or newer!

Verify that Python is running with:
```
>>> python --version
```
The output should return the version of the Python libraries installed in your system. Verify also that the package installer for Python, [PIP](https://pip.pypa.io/en/stable/installation/) is installed.

You'll need an Nvidia API Key in order to use their hosted LLMs. You can create your own API Key here:

<<<TODO>>>

### Installing YARS

First, download the repository as a [ZIP file](https://github.com/apolo74/YARS/archive/refs/heads/main.zip) or (assuming you have already installed the github package) just open a terminal and `git clone` it. Go inside the **YARS** folder and I recommend to work under a virtual environment:
```
>>> python -m venv .venv
```
All the required dependencies are listed inside the *requirements.txt* file. To install them just run:
```
>>> python -m pip install -r requirements.txt
```

## Execution
To start interacting with your PDFs just run the following line. The script accepts a path to a single PDF file or path to a folder with multiple PDF files. Follow the instructions and enjoy!
```
>>> python main_yars.py -d [path_to_pdf_file | path_to_PDFs_folder]
```
