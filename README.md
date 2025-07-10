The best path weights are stored in Baidu Cloud Drive and can be obtained via the following 
link: [https://pan.baidu.com/s/1Nt8gz8JVAYg9PdBNIJahow](https://pan.baidu.com/s/1Nt8gz8JVAYg9PdBNIJahow)  
Extraction code: 1234

Environment

conda create -n SnowNet python=3.7

conda activate SnowNet

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117


Install dependencies

pip install -r requirements.txt


Test

python map-rsod.py

python map-sw.py

Note: You need to download and save the best weights to the folder in advance.
