# AIRi-TV

BETA version of the research dissemination system for AIRi. It's made by a Mixture of Models to generate a conversation based on a research paper, with just 5s of author speech reference, a picture with the author and the article itself.

Made with ❤️ for the Artificial Intelligence Research Institute at the Technical University of Cluj-Napoca.

## SETUP:

IMPORTANT:
A 48GB GPU is required.
Before you continue, make sure you link your HuggingFace account with the machine you work with and ffmpeg version==4.4.2 installed into your docker container. To use Llama-3.1-8B-Instruct (as used in this repo), request access from Meta for the model checkpoint [here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).


### To create all the venvs and meet the requirements of each model and the bash script that does the full generation:


```
git clone https://github.com/davidcombei/AIRi-TV.git
cd AIRi-TV/
mkdir assets/audio/
mkdir assets/video/

conda create -yn  backgrounds python==3.11.14
conda activate backgrounds
pip install rembg onnxruntime "numpy<2.0" torch
conda deactivate backgrounds

conda create -yn llama python==3.11.14
conda activate llama
pip install -r requirements_LLM.txt
conda deactivate 

conda create -yn chatterbox python==3.11
conda activate chatterbox
pip install -r requirements_TTS.txt
pip install chatterbox-tts --no-deps
conda deactivate 

conda create -yn sadtalker python==3.8
conda activate sadtalker
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install -y ffmpeg==4.4.2
pip install -r requirements_VisualModel.txt
conda deactivate 
```

After your venvs are created and ready, you need to get the checkpoints for SadTalker and the enhancer (gfpgan) from [here](https://drive.google.com/drive/folders/1loOFWGCYoBdCn1lRXPqNlBrI3VOcKjGV?usp=sharing).
NOTE: I do not own these checkpoints, these are made in this [Github Repo](https://github.com/OpenTalker/SadTalker)
Move both directories into your cloned git repo and run:
```
mv checkpoints/ VisualModel/SadTalker/
```
Upload your images, audios, and article into `assets/` directory.

Note: you need to upload audio and portrait image for both anchor and author alongside the background and logo pictures. The names for anchor files must be: `anchor.jpg`, `anchor.wav`, and for the background and logo they should be : `background.jpg` and `airi_logo.jpeg`. You can skip the background and logo part by commenting the last part in the bash script.

Final step, run:
```
./run_dissemination.sh assets/your_article.pdf assets/your_image.png assets/your_audio.wav
```
Wait for the system to create your video. The final output will be saved as: `assets/video/articleName_podcast.mp4`.

Enjoy! :)

## Contact

david.combei[at]cs.utcluj.ro

## Acknowledgement

This work was funded by a research scholarship from Bitdefender and by the Romanian
Ministry of Research, Innovation and Digitization project DLT AI SECSPP (id: PN-IV-P6-6.3-SOL-2024-2-0312)

