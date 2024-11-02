# automatic-album-sequencer

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project takes a set of music files and automatically orders them so that they are arranged to approximate how a professional might sequence them. Two approaches are demonstrated here. The first is the direct transformer-based approach presented in *Automatic Album Sequencing* by Vincent Herrmann, Dylan R. Ashley, and Juergen Schmidhuber. The second is the contrastive template-based approach presented in *On the Distillation of Stories for Transferring Narrative Arcs in Collections of Independent Media* by Dylan R. Ashley, Vincent Herrmann, Zachary Friggstad, and Juergen Schmidhuber.

Note that this project is forked from `https://github.com/dylanashley/story-distiller` and contains a lot of material directly copied from there.


## Installation

This project is implemented in [Python](https://www.python.org/) and uses models learned with [PyTorch](https://pytorch.org).

Before installation, first, ensure you have a recent version of Python (>=3.10) and pip, then clone the repository:
```bash
git clone git@github.com:dylanashley/automatic-album-sequencing.git
```

Afterwards, install the Python dependencies using pip:
```bash
pip install -r requirements.txt
```

To use the full suite of file types supported by the tool, it is also necessary to separately install the [ffmpeg](https://ffmpeg.org/) tool by following the instructions for your operating system provided on [their website](https://ffmpeg.org/download.html).

You can now run the web app by directly executing the `app.py` file, run the tool by directly executing the `__main__.py` file, or—if you're using Linux or macOS—you can use the makefile to compile the tool and then install it as an executable Python zip archive:
```bash
make all
sudo make install
```


## App Usage

To run the web app, simply execute the `app.py` file:
```bash
./app.py
```


## Command-line Tool Usage

To run the program, execute it while passing the audio files as command-line arguments:
```bash
sdistil files [files ...] >> playlist.txt
```

[librosa](https://librosa.org/doc/latest/index.html) is used to process the audio files, so most common audio file types are supported.

If you want to try out a different template, pass the `-t` argument to the program with the template file as an argument. Several learned templates are included in the templates directory:

![templates.jpg](https://github.com/dylanashley/automatic-album-sequencing/blob/main/templates.jpg?raw=true)
